# 07 - Local/Container Model Backends (No OpenAI Credits)

## 1. Goal
Run this project against local or self-hosted OpenAI-compatible endpoints so you can avoid hosted API credit usage.

## 2. What Was Added in Code
- New env wiring for OpenAI-compatible base URLs:
  - `MMAA_LLM_BASE_URL`
  - `MMAA_RAG_OPENAI_BASE_URL`
  - `MMAA_RAG_MULTIMODAL_OPENAI_BASE_URL`
  - `MMAA_RAG_RERANKER_OPENAI_BASE_URL`
  - `MMAA_MULTIMODAL_BASE_URL`
- All current OpenAI adapters now accept either:
  - API key (`*_API_KEY`) for hosted providers, or
  - Base URL (`*_BASE_URL`) for local servers.
- Local file image URIs (`file://...`) are converted to `data:` URLs in the vision client for better compatibility with OpenAI-style VLM APIs.
- Webpage URLs passed to vision preprocessing are resolved to concrete image assets when discoverable (meta tags/image links), then sent as image payloads.
- Video pipeline (M4.1) uses strict decoded-frame extraction for per-frame VLM analysis (`opencv-python` required) and fails explicitly if decode is unavailable.
- RAG ingestion is now VL-first for video: it indexes direct `video_url` multimodal embeddings first, with frame analysis as optional enrichment.
- Video ingestion can also append speech transcript evidence (Whisper local ASR) so spoken content becomes retrievable text chunks.

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

MMAA_RAG_MULTIMODAL_ENABLED=true
MMAA_RAG_MULTIMODAL_EMBEDDING_PROVIDER=qwen3_vl
MMAA_RAG_MULTIMODAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
MMAA_RAG_MULTIMODAL_OPENAI_BASE_URL=http://localhost:8001/v1
MMAA_RAG_MULTIMODAL_OPENAI_API_KEY=local-placeholder
MMAA_RAG_MULTIMODAL_EMBEDDING_ENDPOINT=/embeddings

MMAA_RAG_RERANKER_ENABLED=true
MMAA_RAG_RERANKER_PROVIDER=qwen3_vl
MMAA_RAG_RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B
MMAA_RAG_RERANKER_OPENAI_BASE_URL=http://localhost:8002/v1
MMAA_RAG_RERANKER_OPENAI_API_KEY=local-placeholder
MMAA_RAG_RERANKER_QWEN3_VL_ENDPOINT=/rerank

MMAA_MULTIMODAL_PROVIDER=openai
MMAA_MULTIMODAL_VISION_MODEL=qwen3-vl:2b
MMAA_MULTIMODAL_BASE_URL=http://localhost:11434/v1
MMAA_MULTIMODAL_API_KEY=local-placeholder
```

4. Restart backend after env changes.

Note: true VL retrieval requires a Qwen3-VL embedding/reranker-serving endpoint (for example vLLM/qwen-vl-embedding server) at `:8001`; Ollama stays useful for LLM generation and vision analysis.

Optional ingestion enrichment toggle:
- `MMAA_MULTIMODAL_VIDEO_INGEST_ENRICH_WITH_ANALYSIS=false` (default): no frame decode during RAG indexing, relies on direct VL video embeddings.
- Set `true` to append frame/timeline analysis during ingestion (requires `opencv-python` decode support).
- `MMAA_MULTIMODAL_VIDEO_AUDIO_TRANSCRIPTION_ENABLED=true`: transcribe local video audio during ingestion and append timestamped transcript lines (requires `whisper` package + `ffmpeg` in `PATH`).
- `MMAA_MULTIMODAL_VIDEO_AUDIO_TRANSCRIPTION_MODEL=tiny` (or `base`, `small`, etc.): local ASR model size/quality tradeoff.

## 6. vLLM Quick Start (True VL Retrieval)
Example embedding server launch:
```powershell
docker run --runtime nvidia --gpus all `
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface `
  -p 8001:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen3-VL-Embedding-2B
```

Example reranker server launch (separate port):
```powershell
docker run --runtime nvidia --gpus all `
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface `
  -p 8002:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen3-VL-Reranker-2B
```

Then point this app to:
- `MMAA_RAG_MULTIMODAL_OPENAI_BASE_URL=http://localhost:8001/v1`
- `MMAA_RAG_RERANKER_OPENAI_BASE_URL=http://localhost:8002/v1`

## 7. Notes for Resource Budget
- On this hardware, start with `qwen3:4b` for text and `qwen3-vl:2b` for vision/video tasks.
- If latency or VRAM pressure is high, drop text model to `qwen3:1.7b` before reducing retrieval quality knobs.
- Keep `top_k` and generation lengths moderate for stable latency.
- If VRAM pressure appears, reduce model size first, then reduce context/token budgets.
- For decoded-frame video analysis, reduce sampled FPS/max frames first before increasing model size.
