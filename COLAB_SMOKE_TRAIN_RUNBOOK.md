# COLAB SMOKE TRAIN RUNBOOK

Date: 2026-05-16
Purpose: verify that QLoRA fine-tuning runs end-to-end in Colab before spending time on full training

## 1. What this run is for

This smoke run is only a pipeline check.

It is meant to answer:

- does the script run in Colab,
- does the model load in 4-bit,
- does training finish,
- does adapter save work,
- does merge work,
- does inference return a usable answer.

It is not meant to judge final model quality.

## 2. Files to prepare

Upload these files to Colab:

- [finetune_colab.py](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/finetune_colab.py>)
- [data/golf_fitting_train_smoke.jsonl](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/data/golf_fitting_train_smoke.jsonl>)
- optional: [EVAL_PROMPTS.md](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/EVAL_PROMPTS.md>)

Do not upload `gold_set_50.jsonl` as training data.
That file is for review and evaluation.

## 3. Colab runtime setting

In Colab:

1. open a new notebook
2. go to `Runtime > Change runtime type`
3. set `Hardware accelerator = GPU`
4. save

T4 is enough for smoke.
A100 is faster but not required.

## 4. Cell 1: install packages

Paste this first:

```python
!pip install -q -U transformers peft trl bitsandbytes datasets accelerate sentencepiece
```

## 5. Cell 2: upload files

Paste this next:

```python
from google.colab import files

uploaded = files.upload()
print("Uploaded files:", list(uploaded.keys()))
```

Expected upload set:

- `finetune_colab.py`
- `golf_fitting_train_smoke.jsonl`

## 6. Cell 3: create a smoke-config copy of the training script

This avoids editing the original file by hand.

```python
from pathlib import Path

src_path = Path("/content/finetune_colab.py")
dst_path = Path("/content/finetune_colab_smoke.py")

src = src_path.read_text(encoding="utf-8")

replacements = {
    'DATASET_PATH = "data/golf_fitting_train.jsonl"': 'DATASET_PATH = "/content/golf_fitting_train_smoke.jsonl"',
    'OUTPUT_ROOT = Path("./outputs")': 'OUTPUT_ROOT = Path("/content/outputs_smoke")',
    'NUM_EPOCHS = 3': 'NUM_EPOCHS = 1',
    'GRADIENT_ACCUMULATION_STEPS = 8': 'GRADIENT_ACCUMULATION_STEPS = 4',
    'MAX_SEQ_LENGTH = 1536': 'MAX_SEQ_LENGTH = 1024',
    'SAVE_STEPS = 200': 'SAVE_STEPS = 20',
}

for old, new in replacements.items():
    if old not in src:
        raise ValueError(f"Expected text not found: {old}")
    src = src.replace(old, new)

dst_path.write_text(src, encoding="utf-8")
print("Created:", dst_path)
print(dst_path.read_text(encoding="utf-8")[:1200])
```

## 7. Cell 4: run the smoke train

```python
exec(open("/content/finetune_colab_smoke.py", encoding="utf-8").read(), globals())
```

## 8. What success looks like

You should see all of these:

- tokenizer loads
- 4-bit base model loads
- dataset row count prints
- training starts and finishes
- adapter save completes
- merged model save completes
- quick inference test prints an answer

Expected smoke behavior:

- very small dataset
- very short training
- quality is not the goal
- end-to-end completion is the goal

## 9. Cell 5: zip and download smoke outputs

```python
import shutil
from google.colab import files

shutil.make_archive("/content/fitted-golf-smoke-outputs", "zip", "/content/outputs_smoke")
files.download("/content/fitted-golf-smoke-outputs.zip")
```

## 10. Common failure fixes

### CUDA OOM

If GPU memory fails, patch these more aggressively:

```python
from pathlib import Path

p = Path("/content/finetune_colab_smoke.py")
src = p.read_text(encoding="utf-8")
src = src.replace("PER_DEVICE_BATCH_SIZE = 2", "PER_DEVICE_BATCH_SIZE = 1")
src = src.replace("GRADIENT_ACCUMULATION_STEPS = 4", "GRADIENT_ACCUMULATION_STEPS = 8")
src = src.replace("MAX_SEQ_LENGTH = 1024", "MAX_SEQ_LENGTH = 768")
p.write_text(src, encoding="utf-8")
print("OOM-safe config applied")
```

### Runtime disconnected

Do this:

- reconnect the runtime
- upload the two files again
- rerun from Cell 1

### Strange Korean display

If Colab output looks odd, first check whether:

- the generated file paths exist,
- the final inference still returns Korean text,
- the saved model folders were created.

Display issues are less important than whether training actually completed.

## 11. After smoke succeeds

Do this next:

1. keep the smoke outputs for reference
2. review [data/gold_set_50.jsonl](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/data/gold_set_50.jsonl>)
3. score outputs with [EVAL_PROMPTS.md](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/EVAL_PROMPTS.md>)
4. only then move to the full dataset run

## 12. Full train switch

When you are ready for the real run:

- upload `golf_fitting_train.jsonl`
- point `DATASET_PATH` to the full file
- restore:
  - `NUM_EPOCHS = 3`
  - `GRADIENT_ACCUMULATION_STEPS = 8`
  - `MAX_SEQ_LENGTH = 1536`
  - `SAVE_STEPS = 200`
- use a separate output folder such as `/content/outputs_full`

## 13. Important note about the files

Use the files like this:

- `golf_fitting_train_smoke.jsonl`: pipeline smoke training
- `golf_fitting_train.jsonl`: full training
- `gold_set_50.jsonl`: manual review and evaluation only
- `EVAL_PROMPTS.md`: fixed before/after comparison prompts
