# 05 - Progress Log Handoff

## 1. Snapshot
- Date: 2026-03-08
- Branch: `main`
- Current HEAD: working tree includes M2.4 implementation slice (persistent chat + scoped retrieval + transcript enrichment)
- Working tree status: implementation and docs updates pending commit
- Last full test run:
  - `.\venv\Scripts\python.exe -m pytest -q` -> `81 passed`

## 2. Milestone Status
- M0: complete.
- M1: complete and extended.
- M2: complete (LangGraph orchestration + tool discovery endpoint).
- M2.2: complete (Streamlit architecture + implementation flow, route playground).
- M2.3: complete (runtime event stream + status endpoint + bounded revision loop + frontend status/events panel).
- M2.4: complete (durable multi-chat sessions + chat-scoped retrieval + transcript-grade runtime timeline).
- M3: complete and extended (vision preprocess/adapter/fusion + webpage image resolution).
- M4.1: complete.
- M5: planned.
- M5.1 (context compaction): in progress (vertical slice implemented).
- M5.2 (steering controls): in progress (baseline implemented).
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

### 3.7 M5.1 Context Compaction Vertical Slice
- Added context compaction schema + runtime modules:
  - `app/contracts/context.py`
  - `app/core/context_compaction.py`
  - `app/agents/context_manager.py`
- Integrated orchestrator pre-step memory guard:
  - `app/agents/orchestrator.py` now applies context compaction under token pressure before each graph pass.
  - Emits runtime event with `reason_code=context_compaction` when triggered.
- State/config wiring:
  - `app/agents/state.py` now stores durable `context_checkpoints`.
  - `app/core/config.py` adds compaction settings (`context_compaction_*`).
  - `app/core/dependencies.py` wires context manager into orchestrator.
- Added regression tests:
  - `tests/test_m51_context_compaction.py`

### 3.8 Frontend Chat Workspace Extension (Implemented)
- Added dedicated `Chat` tab in `frontend/streamlit_app.py` for multi-turn conversations.
- Each chat turn supports:
  - mode selection (`RAG` / `agentic` / auto),
  - optional agent tool multiselect,
  - source ingestion via URIs, file upload, and clipboard image paste.
- Agentic turns use compact live run status updates (M2.3 runtime stream/status integration).
- Chat flow composes recent turns into contextual query payloads and remains compatible with backend compaction (M5.1) and steering (M5.2) controls.

### 3.9 Research + Blueprint for Durable Chat and Rich Runtime Transcript
- Completed an industry-backed research pass (as of 2026-03-08) and documented implementation blueprint in:
  - `docs/12-chat-persistence-and-runtime-reasoning-plan.md`
- Scope defined for next implementation slice:
  - persistent chat sessions and server-side message/file storage,
  - chat-scoped retrieval metadata filters,
  - scrollable runtime transcript with stage/evidence/tool detail payloads,
  - provider-aware reasoning rendering (reasoning channel when exposed; structured execution details otherwise).

### 3.10 M2.4 Implementation Delivered
- Added durable chat contracts/store/routes:
  - `app/contracts/chat.py`
  - `app/storage/chat_store.py`
  - `app/api/routes/chat.py`
  - `app/api/router.py` now mounts chat routes.
- Added retrieval scoping support:
  - `app/interfaces/vector_store.py`, `app/interfaces/retriever.py`
  - `app/rag/ingestion.py`, `app/rag/retriever.py`
  - `app/storage/pgvector_store.py`, `app/storage/qdrant_store.py`, `app/storage/fallback_vector_store.py`
  - `app/contracts/schemas.py` now supports request-level metadata filters.
- Runtime transcript enrichment:
  - `app/agents/orchestrator.py`, `app/agents/answer_agent.py`, `app/core/event_bus.py`
  - step events now include evidence/analysis/answer previews and normalized `event_family` metadata.
- Frontend chat upgraded to persistent multi-chat:
  - `frontend/streamlit_app.py` now uses backend chat sessions/messages and live transcript updates during agentic runs.
- Regression tests added:
  - `tests/test_m24_chat_sessions.py`
  - `tests/test_m24_chat_scoped_retrieval.py`
  - `tests/test_m24_runtime_transcript.py`

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
- `docs/11-live-status-and-loop-orchestration-plan.md`
- `docs/12-chat-persistence-and-runtime-reasoning-plan.md`
- `frontend/architecture.py`

## 6. Known Caveats
- Video frame extraction is strict decode-based (`cv2` required) and fails explicitly when decode backend is unavailable.
- FastAPI startup deprecation warning (`on_event`) remains non-blocking.
- Full-suite benchmark/eval artifacts remain M6 scope.

## 7. Suggested Immediate Next Step
1. Start M5.3 (multimodal embedding stack) in a separate branch with named-vector contract and adapter tests.
2. Define migration strategy for chat store from SQLite to Postgres for multi-user deployment.
3. Add pagination/retention controls for long chat transcripts and runtime events.
