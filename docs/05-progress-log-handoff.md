# 05 - Progress Log Handoff

## 1. Snapshot
- Date: 2026-03-07
- Branch: `main`
- Current HEAD: `a43ea7f` (multimodal visibility + vision hardening + indexed sources)
- Working tree status: includes M2.3 planning surfaces (live status + loop-safe revision architecture/docs) pending commit
- Last full test run:
  - `.\venv\Scripts\python.exe -m pytest -q` -> `54 passed`

## 2. Milestone Status
- M0: complete.
- M1: complete and extended.
- M2: complete (LangGraph orchestration + tool discovery endpoint).
- M2.2: complete (Streamlit architecture + implementation flow, route playground).
- M2.3: complete (runtime event stream + status endpoint + bounded revision loop + frontend status/events panel).
- M3: complete and extended (vision preprocess/adapter/fusion + webpage image resolution).
- M4.1: complete.
- M5: planned.
- M5.1 (context compaction): planned.
- M5.2 (steering controls): planned.
- M5.3 (multimodal embeddings stack): planned.
- M6+: planned.

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

### 3.5 M4.1 Video Timeline + Frame-Evidence Pipeline
- Upgraded video pipeline:
  - `VideoFrameSampler` now supports optional local decode-based frame extraction and emits frame image payloads.
  - `VideoAnalysisAdapter` now adds per-frame vision analysis before temporal aggregation.
  - `TemporalAggregator` prioritizes frame findings and retains summary fallback.
- `/video/analyze` preserves contract and now reflects frame-evidence-aware timeline composition.
- Added/updated M4 tests:
  - decoded-frame path behavior
  - frame-level finding integration
  - temporal ordering/coherence
  - route-level budget behavior
- Ingestion integration:
  - `DocumentIngestionService` now reuses video adapter pipeline so video timeline evidence is indexed into shared RAG.

### 3.6 Planning Additions (M5.1 + M5.2 + M2.3)
- Added roadmap/requirements entries for:
  - context compaction (Codex-style checkpoint summaries under token pressure)
  - steering controls (style/tool/grounding policy profiles)
  - live runtime status event stream + loop-safe revision graph with deterministic guardrails
- Updated frontend architecture diagram to reflect:
  - M4.1 frame-evidence flow
  - planned runtime controls for compaction and steering
  - planned revision/evaluator loop with explicit infinite-loop prevention boundaries

## 4. Config and Runtime Notes
- `.env` / `.env.example` are organized for Ollama-only OpenAI-compatible local profile.
- Current local model defaults documented around:
  - text model: `qwen3:4b`
  - embedding model: `nomic-embed-text`
  - vision model: `qwen3-vl:2b`

## 5. Documentation Updated in This Pass
- `README.md`
- `docs/01-requirements.md`
- `docs/02-implementation-roadmap.md`
- `docs/03-didactic-traceability.md`
- `docs/04-file-traceability-by-milestone.md`
- `docs/08-multimodal-research-and-m41.md`
- `docs/07-local-model-backends.md`
- `frontend/architecture.py`

## 6. Known Caveats
- Video frame extraction is strict decode-based (`cv2` required) and fails explicitly when decode backend is unavailable.
- FastAPI startup deprecation warning (`on_event`) remains non-blocking.
- Full-suite benchmark/eval artifacts remain M6 scope.

## 7. Suggested Immediate Next Step
1. Start M5.1 context compaction: threshold-triggered checkpoint summaries + pinned-context schema + regression tests.
2. Implement M5.2 steering baseline: request-level policy profile (style/tool/citation/risk controls) with contract tests.
3. Begin M5.3 multimodal embedding stack: multivector payload model + adapter path while preserving current retrieval contract.
