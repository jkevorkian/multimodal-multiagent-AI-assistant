# 05 - Progress Log Handoff

## 1. Snapshot
- Date: 2026-03-05
- Branch: `main`
- Current HEAD: `6b98bb0` (pre-commit snapshot in this workspace)
- Working tree status: M2/M2.2/M3 implementation set is present with additional UX and ingestion upgrades pending commit
- Last full test run:
  - `.\venv\Scripts\python.exe -m pytest -q` -> `48 passed`

## 2. Milestone Status
- M0: complete.
- M1: complete and extended.
- M2: complete (LangGraph orchestration + tool discovery endpoint).
- M2.2: complete (Streamlit architecture + implementation flow, route playground).
- M3: complete and extended (vision preprocess/adapter/fusion + webpage image resolution).
- M4+: planned.

## 3. Delivered Since Previous Handoff

### 3.1 Ingestion and File Type Expansion
- Added/validated broader document ingestion support:
  - markdown/text-like formats
  - `docx`, `pptx`, `xlsx`
  - html extraction path
- Relevant modules/tests:
  - `app/rag/ingestion.py`
  - `tests/test_m1_rag.py`

### 3.2 Agent UX and Tool Discoverability
- Added `GET /agents/tools` for catalog-style tool discovery.
- `/agents/run` behavior clarified and preserved: if `tools` omitted, backend enables all registered tools.
- Streamlit moved from manual CSV tool typing to tool multiselect populated from API catalog.
- Relevant modules/tests:
  - `app/api/routes/agents.py`
  - `app/tools/registry.py`
  - `app/contracts/schemas.py`
  - `frontend/streamlit_app.py`
  - `tests/test_api_contracts.py`

### 3.3 Vision Robustness for Real URLs
- Vision preprocessing now resolves webpage URLs (HTML pages) to concrete image assets when possible (`og:image`, `twitter:image`, and image links), then converts to model-ready payload.
- Invalid pages without discoverable images now fail validation cleanly.
- Relevant modules/tests:
  - `app/vision/preprocess.py`
  - `tests/test_m3_vision.py`

### 3.4 Frontend Ingestion UX Upgrade
- Added clipboard image paste support (`Ctrl+V`) in Streamlit ingestion flows (Implementation + Ingest tabs) via `streamlit-paste-button`.
- Pasted images are persisted to temp and added as `file://` sources for backend ingest.
- Relevant modules:
  - `frontend/streamlit_app.py`
  - `requirements.txt`
  - `README.md`

## 4. Config and Runtime Notes
- `.env` / `.env.example` are organized for Ollama-only OpenAI-compatible local profile.
- Current local model defaults documented around:
  - text model: `qwen3:4b`
  - embedding model: `nomic-embed-text`
  - vision model: `llava:7b`

## 5. Documentation Updated in This Pass
- `README.md`
- `docs/01-requirements.md`
- `docs/02-implementation-roadmap.md`
- `docs/03-didactic-traceability.md`
- `docs/04-file-traceability-by-milestone.md`
- `docs/07-local-model-backends.md`

## 6. Known Caveats
- Video endpoint is still heuristic-first in current scope; full temporal reasoning remains M4.
- FastAPI startup deprecation warning (`on_event`) remains non-blocking.
- Full-suite benchmark/eval artifacts remain M6 scope.

## 7. Suggested Immediate Next Step
1. Commit the current integrated set with a milestone-aware message.
2. Then plan M4 scope slice (frame sampler + temporal aggregation + tests) before entering M5 hardening/evaluation work.
