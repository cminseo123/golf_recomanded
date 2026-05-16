"""
FITTED Qwen2.5-3B-Instruct + QLoRA training script for Google Colab.

Expected flow:
1. Install packages
2. Load chat-format JSONL dataset
3. Fine-tune with QLoRA (4-bit)
4. Save LoRA adapter
5. Merge adapter into a standalone HF model
6. Run a quick inference test
7. Convert merged model to GGUF later

Target runtime guidance:
- T4 16GB: about 2~3 hours
- A100 40GB: about 40 minutes
"""

from __future__ import annotations

# %% 1. Install dependencies in Colab
# !pip install -q -U transformers peft trl bitsandbytes datasets accelerate sentencepiece

# %% 2. Imports
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# %% 3. User config
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "data/golf_fitting_train.jsonl"

OUTPUT_ROOT = Path("./outputs")
ADAPTER_DIR = OUTPUT_ROOT / "fitted-golf-qwen25-3b-lora"
MERGED_DIR = OUTPUT_ROOT / "fitted-golf-qwen25-3b-merged"

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1536
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LOGGING_STEPS = 20
SAVE_STEPS = 200

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# %% 4. Helpers
def get_compute_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def print_gpu_summary(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(f"{prefix}: CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{prefix}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %% 5. Load tokenizer and quantized base model
compute_dtype = get_compute_dtype()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading 4-bit base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
print_gpu_summary("After base model load")


# %% 6. Attach LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# %% 7. Load and format dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded dataset rows: {len(dataset)}")


def format_to_chat_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = dataset.map(format_to_chat_text, remove_columns=dataset.column_names)
print("Dataset preview:")
print(dataset[0]["text"][:600])


# %% 8. Trainer config
training_args = TrainingArguments(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    bf16=(compute_dtype == torch.bfloat16),
    fp16=(compute_dtype == torch.float16),
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)


# %% 9. Train
print("Starting QLoRA fine-tuning...")
print("Expected runtime: T4 16GB ~2~3h / A100 40GB ~40m")
trainer.train()
print("Training complete.")
print_gpu_summary("After training")


# %% 10. Save LoRA adapter
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Saved LoRA adapter to: {ADAPTER_DIR}")


# %% 11. Merge adapter into a standalone HF model
print("Reloading base model in fp16/bf16 for merge...")
del trainer
del model
clear_memory()

merge_dtype = get_compute_dtype()
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=merge_dtype,
    device_map="auto",
    trust_remote_code=True,
)
merged_model = PeftModel.from_pretrained(base_model_for_merge, ADAPTER_DIR)
merged_model = merged_model.merge_and_unload()

MERGED_DIR.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Saved merged model to: {MERGED_DIR}")
print_gpu_summary("After merge")


# %% 12. Quick inference test
def run_test_inference(user_prompt: str, max_new_tokens: int = 320) -> str:
    messages = [
        {
            "role": "system",
            "content": "당신은 20년 경력의 전문 골프 클럽 피터입니다. 한국어로 답변하세요.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    target_device = next(merged_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


test_prompt = """
다음 골퍼 데이터를 보고 피팅 의견을 주세요.
- 성별/나이: 남성 / 43세
- 핸디캡: 18
- 헤드 스피드: 92mph
- 볼 스피드: 135mph
- 발사각: 11.5도
- 스핀: 3200rpm
- 어택 앵글: -1.5도
- 구질: 슬라이스
- 스윙 패스 경향: 아웃투인
- 고민 사항: 슬라이스, 거리 부족
"""

print("=== Inference test ===")
print(run_test_inference(test_prompt))


# %% 13. GGUF conversion notes
# Run these in Colab or in a separate environment after training.
#
# 13-1. Clone llama.cpp
# !git clone https://github.com/ggerganov/llama.cpp.git
#
# 13-2. Convert merged HF model to f16 GGUF
# !python llama.cpp/convert_hf_to_gguf.py ./outputs/fitted-golf-qwen25-3b-merged --outfile fitted-golf-qwen25-3b-f16.gguf --outtype f16
#
# 13-3. Quantize to q4_k_m
# !cd llama.cpp && cmake -B build && cmake --build build --config Release
# !./llama.cpp/build/bin/llama-quantize fitted-golf-qwen25-3b-f16.gguf fitted-golf-qwen25-3b-q4_k_m.gguf q4_k_m
#
# 13-4. Example Ollama Modelfile
# FROM ./fitted-golf-qwen25-3b-q4_k_m.gguf
# SYSTEM "당신은 20년 경력의 전문 골프 클럽 피터입니다. 한국어로 간결하고 구체적으로 답변하세요."
#
# 13-5. Create the Ollama model
# !ollama create fitted-golf -f Modelfile
