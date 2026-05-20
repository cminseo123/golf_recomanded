# QUICK RUN FIRST

Date: 2026-05-16
Purpose: get FITTED running before any full training work

## 1. Fastest path

If the goal is just "make it run first", use one of these two modes.

### Mode A: fallback-first demo

No model setup required.

```powershell
powershell -ExecutionPolicy Bypass -File .\start_local_demo.ps1
```

Then open:

```text
http://127.0.0.1:4173
```

What you get:

- the normal fitting flow,
- AI summary card,
- AI chat panel,
- fallback heuristic explanations if no local model is available.

### Mode B: base Ollama model demo

If you already have Ollama or want a quick base model test without training:

1. Start Ollama:

```powershell
ollama serve
```

2. Pull a small model once:

```powershell
ollama pull qwen2.5:3b
```

3. Start the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_local_demo.ps1 -UseOllama -Model qwen2.5:3b
```

Then open:

```text
http://127.0.0.1:4173
```

What you get:

- the same fitting UI,
- Ollama-backed AI explanation and chat if the model responds,
- automatic fallback if the named model is missing or Ollama is down.

## 2. Model auto-try behavior

The local server now tries these models in order before dropping to fallback:

- `fitted-golf`
- `qwen2.5:3b-instruct`
- `qwen2.5:3b`
- `qwen2.5:1.5b-instruct`
- `qwen2.5:1.5b`

That means you do not need a trained model yet.

## 3. Health check

You can verify the backend quickly:

```text
http://127.0.0.1:4173/api/health
```

You should see:

- `ok: true`
- the Ollama URL
- the primary model
- the candidate model list

## 4. What to look for in the UI

After running a fitting:

- if the AI card shows `OLLAMA · <model>`, the local model answered
- if it shows `FALLBACK`, the app still works without a model
- if it shows `OFFLINE`, the backend is not running

## 5. Recommended order right now

Use this order before any full training:

1. run Mode A once and confirm the app works end-to-end
2. if Ollama is available, run Mode B once with `qwen2.5:3b`
3. only after that, return to smoke training and full training

## 6. Relevant files

- launcher: [start_local_demo.ps1](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/start_local_demo.ps1>)
- backend: [server.js](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/server.js>)
- frontend: [index.html](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/index.html>)
- detailed setup: [LOCAL_MVP_SETUP.md](</C:/Users/USER/Desktop/골프 클럽 장비 추천 Saas/LOCAL_MVP_SETUP.md>)
