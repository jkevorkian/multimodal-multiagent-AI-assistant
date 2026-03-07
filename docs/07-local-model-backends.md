# 07 - Local/Container Model Backends (No OpenAI Credits)

## 1. Goal
Run this project against local or self-hosted OpenAI-compatible endpoints so you can avoid hosted API credit usage.

## 2. What Was Added in Code
- New env wiring for OpenAI-compatible base URLs:
  - `MMAA_LLM_BASE_URL`
  - `MMAA_RAG_OPENAI_BASE_URL`
  - `MMAA_MULTIMODAL_BASE_URL`
- All current OpenAI adapters now accept either:
  - API key (`*_API_KEY`) for hosted providers, or
  - Base URL (`*_BASE_URL`) for local servers.
- Local file image URIs (`file://...`) are converted to `data:` URLs in the vision client for better compatibility with OpenAI-style VLM APIs.
- Webpage URLs passed to vision preprocessing are resolved to concrete image assets when discoverable (meta tags/image links), then sent as image payloads.
- Video pipeline (M4.1) uses strict decoded-frame extraction for per-frame VLM analysis (`opencv-python` required) and fails explicitly if decode is unavailable.

## 3. Recommended Option for This Laptop (RTX 4070M, 8 GB VRAM)
Use **Ollama** as the first local profile.

Why:
- Easiest setup and model management.
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`).
- Good small/medium models that fit laptop constraints.

## 4. Option Matrix
1. Ollama (recommended start)
- Best for: fastest path to local inference with minimal ops overhead.
- API style: OpenAI-compatible.
- Container-friendly: yes.

2. vLLM
- Best for: higher serving throughput and advanced scheduling features.
- API style: OpenAI-compatible.
- Container-friendly: yes.

3. SentenceTransformers (already in repo)
- Best for: fully local neural embeddings in-process without an embeddings API server.
- API style: Python in-process, not HTTP.

## 5. Ollama Container Setup
1. Start Ollama with Docker Compose (CPU mode):
```powershell
docker compose -f deployment/docker-compose.ollama.yml up -d
```

1b. Start Ollama with Docker Compose (GPU mode):
```powershell
docker compose -f deployment/docker-compose.ollama.yml -f deployment/docker-compose.ollama.gpu.yml up -d
```

1c. Verify processor selection:
```powershell
docker exec -it mm_maa_ollama ollama ps
```

2. Pull models (examples):
```powershell
docker exec -it mm_maa_ollama ollama pull qwen3:4b
docker exec -it mm_maa_ollama ollama pull qwen3-vl:2b
docker exec -it mm_maa_ollama ollama pull nomic-embed-text
```

3. Configure this app (`.env`):
```dotenv
MMAA_LLM_PROVIDER=openai
MMAA_LLM_MODEL=qwen3:4b
MMAA_LLM_BASE_URL=http://localhost:11434/v1
MMAA_LLM_API_KEY=local-placeholder

MMAA_RAG_EMBEDDING_PROVIDER=openai
MMAA_RAG_EMBEDDING_MODEL=nomic-embed-text
MMAA_RAG_OPENAI_BASE_URL=http://localhost:11434/v1
MMAA_RAG_OPENAI_API_KEY=local-placeholder

MMAA_MULTIMODAL_PROVIDER=openai
MMAA_MULTIMODAL_VISION_MODEL=qwen3-vl:2b
MMAA_MULTIMODAL_BASE_URL=http://localhost:11434/v1
MMAA_MULTIMODAL_API_KEY=local-placeholder
```

4. Restart backend after env changes.

## 6. vLLM Quick Start (Alternative)
Example GPU container launch:
```powershell
docker run --runtime nvidia --gpus all `
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface `
  -p 8001:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model NousResearch/Meta-Llama-3-8B-Instruct
```

Then point this app to `http://localhost:8001/v1` via `MMAA_LLM_BASE_URL` and/or `MMAA_RAG_OPENAI_BASE_URL`.

## 7. Notes for Resource Budget
- On this hardware, start with `qwen3:4b` for text and `qwen3-vl:2b` for vision/video tasks.
- If latency or VRAM pressure is high, drop text model to `qwen3:1.7b` before reducing retrieval quality knobs.
- Keep `top_k` and generation lengths moderate for stable latency.
- If VRAM pressure appears, reduce model size first, then reduce context/token budgets.
- For decoded-frame video analysis, reduce sampled FPS/max frames first before increasing model size.
