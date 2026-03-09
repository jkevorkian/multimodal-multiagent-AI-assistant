# 02 - Implementation Roadmap

## 1. Roadmap Goal
Implement the assistant in milestone order so each phase is deployable, testable, and didactically traceable.

## 2. Milestone Order Overview
| Milestone | Name | Difficulty | Depends on |
| --- | --- | --- | --- |
| M0 | Skeleton and contracts | Medium | None |
| M1 | Text RAG baseline | Medium | M0 |
| M2 | ReAct tools + multi-agent orchestration | High | M1 |
| M2.2 | Streamlit frontend + architecture visualization | Low | M2 |
| M2.3 | Live run status + loop-safe revision orchestration | Medium | M2.2 |
| M2.4 | Persistent chat sessions + scoped context + runtime transcript | High | M2.3, M5.1, M5.2 |
| M3 | Image multimodal path | Medium | M2.2 |
| M4 | Video MVP path | High | M3 |
| M5 | Production hardening | High | M1-M4 |
| M5.1 | Context compaction (Codex-style) | High | M5 |
| M5.2 | Steering and policy controls | High | M5 |
| M5.3 | Multimodal embedding stack (Qwen3-VL + multivector DB) | High | M4-M5.2 |
| M6 | Evaluation and deployment | Medium | M5-M5.3 |

## 3. Milestone Details

## M0 - Skeleton and Contracts
### Objective
Create a stable architecture shell with provider abstraction, endpoint contracts, and base observability.

### Outputs
- Code modules:
  - `app/main.py`
  - `app/api/routes/*.py`
  - `app/core/config.py`
  - `app/core/logging.py`
  - `app/contracts/schemas.py`
  - `app/interfaces/*.py`
- Endpoints:
  - `/health` returns service status and version.
  - Endpoint stubs for all required routes.
- Tests:
  - Contract tests for route registration and schema validation.
- Docs updates:
  - Initialize technical decision log in didactic doc.

### Risks and Mitigation
- Risk: Over-coupled provider logic in API handlers.
- Mitigation: Enforce interface-based dependency injection from day one.

### Rollback/Fallback
- Fallback to single provider adapter while keeping interface layer unchanged.

### Definition of Done
- All required routes exist and return valid placeholder responses.
- Interfaces compile and are import-resolvable.
- Structured logging is emitted for each request.

## M1 - Text RAG Baseline
### Objective
Deliver ingest -> embed -> retrieve -> answer flow for text queries using modern RAG baseline techniques.

### Outputs
- Code modules:
  - `app/rag/ingestion.py`
  - `app/rag/chunking.py`
  - `app/rag/embeddings.py`
  - `app/rag/retriever.py`
  - `app/rag/reranker.py`
  - `app/storage/pgvector_store.py`
  - `app/storage/qdrant_store.py`
  - `app/storage/fallback_vector_store.py`
  - `app/interfaces/reranker.py`
- Endpoints:
  - `/ingest/documents`
  - `/query`
- Tests:
  - Embedding provider selection tests.
  - Hybrid retrieval + rank-fusion tests.
  - Reranker hook tests.
  - Vector store fallback/coexistence tests (external store primary, pgvector secondary).
  - PDF and URL ingestion tests.
  - Chunking determinism tests.
  - Retrieval smoke tests with known corpus.
- Docs updates:
  - Add neural embedding strategy, hybrid retrieval design, and reranking rationale in didactic doc.

### Risks and Mitigation
- Risk: Latency increase from hybrid retrieval and reranking.
- Mitigation: Configure dense/lexical/rerank candidate budgets and default rerank off for small corpora.

- Risk: provider/runtime mismatch (missing external embedding model dependency).
- Mitigation: keep deterministic fallback provider and explicit startup logging of active provider.

### Rollback/Fallback
- Fallback to deterministic embeddings and lexical-only retrieval if neural provider unavailable.

### Definition of Done
- Query answers include retrieved citations.
- Hybrid retrieval (dense + lexical + rank fusion) works end-to-end.
- Reranker hook is pluggable and test-covered.
- PGVector + JSONB metadata persistence works end-to-end.
- External vector-store adapter can coexist with pgvector/in-memory fallback.

## M2 - ReAct Tools + Multi-Agent Orchestration
### Objective
Add explicit role-based orchestration and tool routing with durable execution and MCP-ready tool interfaces.

### Outputs
- Code modules:
  - `app/agents/state.py`
  - `app/agents/research_agent.py`
  - `app/agents/analyst_agent.py`
  - `app/agents/answer_agent.py`
  - `app/agents/orchestrator.py`
  - `app/tools/registry.py`
  - `app/tools/mcp_adapter.py`
  - `app/agents/checkpoint_store.py`
  - `app/llm/clients.py`
- Endpoints:
  - `/agents/run`
  - `/agents/tools` (tool discovery for frontend/API consumers)
- Tests:
  - State transition tests.
  - Tool-selection scenario tests.
  - Tool catalog contract tests.
  - Tool-call idempotency tests across resume points.
  - Human-interrupt/resume tests for long-running runs.
  - Timeout/retry behavior tests.
- Docs updates:
  - Add agent control-flow trace example, state model explanation, and MCP tool integration pattern.
  - Document default tool behavior when `tools` is omitted and `/agents/tools` discovery endpoint.

### Risks and Mitigation
- Risk: Unstable or looping agent behavior.
- Mitigation: Add max-step guardrails, tool budget, loop detection, and checkpoint-based resume.

### Rollback/Fallback
- Route `/agents/run` to single-agent deterministic chain on orchestration failure.

### Definition of Done
- Multi-agent sequence executes with trace logs.
- Tool calls are bounded and observable.
- Runs can be resumed from checkpoints without repeating completed tool calls.
- Orchestration runtime is backed by LangGraph while preserving existing API contract.
- Answer synthesis uses provider-backed LLM selection (OpenAI when configured, deterministic grounded fallback otherwise).

### Recommended Tool Catalog (M2.x Follow-up)
- `web_search_tool`: fetch recent/public info when retrieval corpus is insufficient.
- `url_fetch_tool`: deterministic fetch + sanitize of specific URLs referenced by user.
- `rag_debug_tool`: inspect retrieved chunks/scores to debug grounding quality.
- `filesystem_tool` (scoped): list/read allowed local workspace files for operator workflows.
- `vision_ocr_tool`: extract text from image frames/screenshots for stronger multimodal retrieval.
- `asr_tool`: transcript extraction from video/audio for timeline-grounded indexing.
- `system_metrics_tool`: expose local GPU/VRAM/runtime health for on-prem tuning.

### Implementation Status
- Current branch status (2026-03-04): implemented and test-covered, with LangGraph orchestration and provider-backed LLM wiring.

## M2.2 - Streamlit Frontend + Architecture Visualization
### Objective
Provide a lightweight UI to exercise backend routes and communicate the high-level system architecture (modules and control/data flows).

### Outputs
- Code modules:
  - `frontend/streamlit_app.py`
  - `frontend/architecture.py`
- Endpoints:
  - no new backend endpoint; reuses existing API routes from `M0-M2`.
- Tests:
  - Frontend architecture helper output test.
- Docs updates:
  - Add M2.2 rationale and traceability entries in didactic and file-level docs.
  - Document frontend usability features (tool multiselect via `/agents/tools`, clipboard image paste for ingest).
  - Add multi-turn Chat tab plan/notes:
    - conversation history with multiple user/assistant turns
    - per-turn source ingestion (URIs, uploads, clipboard images)
    - mode selection (`RAG Query` / `Agentic Run` / auto)
    - optional multi-select tools for agent runs
    - live run status box updates during agent execution
    - compatibility with M5.1 compaction + M5.2 steering controls

### Risks and Mitigation
- Risk: frontend drifts from backend contracts.
- Mitigation: route forms map directly to current Pydantic contracts and are validated by existing API tests.

### Rollback/Fallback
- Fallback to CLI/API-only usage while keeping architecture module as static documentation artifact.

### Definition of Done
- Streamlit app can call core routes (`/health`, `/ingest/documents`, `/query`, `/agents/run`, `/metrics`).
- Architecture diagram renders and explains major module interactions at high level.
- M2.2 frontend helper tests pass.

### Implementation Status
- Current branch status (2026-03-04): implemented in current working tree.
- Current branch status (2026-03-05): extended with tool-discovery UX and clipboard image paste ingestion.
- Current branch status (2026-03-08): chat-tab extension implemented with multi-turn chat, per-turn ingest (URI/upload/clipboard), RAG/agentic mode selection, optional tool multiselect, and live status integration.
- Current branch status (2026-03-08): chat workspace now uses durable backend sessions with persisted messages/files and revisit/continue behavior.

## M2.3 - Live Run Status + Loop-Safe Revision Orchestration
### Objective
Expose high-level runtime progress to end users ("thinking", "processing", "tool running", "revising") and support evaluator/revision loops with deterministic guards that prevent infinite cycles.

### Outputs
- Code modules:
  - `app/contracts/runtime_events.py`
  - `app/core/event_bus.py`
  - `app/agents/loop_controller.py`
  - `app/agents/progress_mapper.py`
- Endpoints:
  - `GET /runs/{run_id}/events` (SSE stream)
  - `GET /runs/{run_id}/status` (latest snapshot)
- Frontend:
  - Live status banner and timeline panel in `frontend/streamlit_app.py`.
  - Agent loop/revision visualization in `frontend/architecture.py`.
- Tests:
  - Event contract serialization tests (ordering + required fields).
  - Loop guard tests (`max_steps`, `max_tool_calls`, stagnation, timeout).
  - SSE reconnection/replay tests (last event id, resume).
  - Agent runtime UX tests (status text transitions by stage).
- Docs updates:
  - Add runtime event taxonomy and state-transition guidance.
  - Add industrial-pattern mapping and implementation blueprint in `docs/11-live-status-and-loop-orchestration-plan.md`.

### Risks and Mitigation
- Risk: noisy event streams overwhelm UI and operators.
- Mitigation: coalesce repeated progress events and throttle non-critical updates.

- Risk: revision loops can still burn budget with no quality gain.
- Mitigation: enforce multi-guard termination (`max_steps`, stagnation detector, tool/time budgets) and emit explicit `guardrail_triggered` reason.

### Rollback/Fallback
- Disable live streaming and keep snapshot-only status endpoint if SSE transport causes instability.

### Definition of Done
- End users can see meaningful run status updates in near real time.
- Agent revisions are observable and bounded by deterministic guardrails.
- Tool usage status is surfaced when tools run (started/completed/failed with latency).
- Failure and guardrail exits are explicit, not silent.

### Implementation Status
- Current branch status (2026-03-08): implemented with runtime event contract/event bus, `/runs/{run_id}/events` SSE + `/runs/{run_id}/status`, bounded revision loop behavior, and frontend runtime timeline/status panel.

## M2.4 - Persistent Chat Sessions + Scoped Context + Runtime Transcript
### Objective
Turn chat from a frontend-only interaction into a durable backend feature with revisitable conversations, per-chat files/context isolation, and a scrollable step/evidence transcript during runs.

### Outputs
- Code modules:
  - `app/contracts/chat.py`
  - `app/storage/chat_store.py`
  - `app/api/routes/chat.py`
  - updates to `app/rag/ingestion.py`, `app/rag/retriever.py`, and vector-store adapters for metadata filtering
  - updates to `app/core/event_bus.py` and runtime event metadata shaping for transcript details
  - updates to `frontend/streamlit_app.py` chat workspace and transcript panel
- Endpoints:
  - `POST /chat/sessions`, `GET /chat/sessions`, `GET/PATCH /chat/sessions/{chat_id}`
  - `POST/GET /chat/sessions/{chat_id}/messages`
  - `POST /chat/sessions/{chat_id}/files`
  - `GET /chat/sessions/{chat_id}/runs/{run_id}/events`
- Features:
  - durable multi-chat history (create/select/revisit/continue)
  - per-chat file persistence and retrieval scoping (`chat_id` metadata filter)
  - cohesive multi-turn context composition
  - live scrollable transcript with stage history + retrieved evidence + tool/result details
  - provider-aware reasoning rendering (explicit reasoning channel when exposed; structured execution trace otherwise)
- Tests:
  - chat-session persistence tests
  - chat-scoped retrieval filter tests
  - runtime transcript payload tests
- Docs updates:
  - implementation/research blueprint in `docs/12-chat-persistence-and-runtime-reasoning-plan.md`

### Risks and Mitigation
- Risk: cross-chat data leakage through retrieval.
- Mitigation: mandatory `chat_id` filter in chat mode and explicit global-scope opt-in.

- Risk: exposing hidden chain-of-thought in unsupported providers.
- Mitigation: render only provider-exposed reasoning fields; fallback to structured execution evidence.

- Risk: storage growth from files/events.
- Mitigation: retention/archival policy per chat and bounded event windows.

### Rollback/Fallback
- Keep existing non-persistent Chat tab path available behind a feature flag until durable routes pass regression tests.

### Definition of Done
- Chats persist across app restarts and can be resumed.
- Uploaded sources are attached to chat scope and retrieval is correctly filtered.
- Transcript panel is scrollable and shows ordered step history with evidence/tool detail payloads.
- Existing `/query`, `/agents/run`, and `/runs/*` routes remain backward compatible.

### Implementation Status
- Current branch status (2026-03-08): implemented.
- Delivered in this slice:
  - durable chat session APIs (`/chat/sessions*`) with persisted messages/files/run links.
  - chat-scoped ingestion metadata tagging and retrieval filtering (`chat_id`, `message_id`, `scope`).
  - runtime transcript enrichment (retrieval preview, analysis trace, answer preview, tool/error details).
  - Streamlit Chat tab migrated from session-local state to backend-persisted multi-chat workflow.
  - regression tests for session persistence, chat-scoped retrieval isolation, and transcript event detail payloads.

## M3 - Image Multimodal Path
### Objective
Support image analysis and integrate visual findings into answer generation with evidence-grounded multimodal retrieval.

### Outputs
- Code modules:
  - `app/multimodal/clients.py`
  - `app/rag/ingestion.py` (multimodal ingestion integration)
  - `app/vision/preprocess.py`
  - `app/vision/adapter.py`
  - `app/vision/fusion.py`
- Endpoints:
  - `/vision/analyze`
- Tests:
  - Deterministic image fixtures.
  - Vision-grounding tests that require explicit visual evidence references.
  - Request size/type validation tests.
- Docs updates:
  - Add vision pipeline didactic section and modality-fusion strategy notes.

### Risks and Mitigation
- Risk: Hallucinated visual details.
- Mitigation: Force evidence-grounded response format with confidence notes.

### Rollback/Fallback
- If vision model unavailable, return explicit degraded-mode message and continue text flow.

### Definition of Done
- Image endpoint returns stable structured response with confidence and evidence summary.

### Implementation Status
- Current branch status (2026-03-04): implemented in current working tree and test-covered.
- Delivered now:
  - image/video source ingestion into shared vector pipeline with modality metadata
  - vision preprocessing/adapter/fusion modules
  - evidence-grounded `/vision/analyze` responses with validation checks
  - webpage URL image resolution in preprocessor (meta/image extraction -> concrete image inference payload)
  - OpenAI-compatible local endpoint routing (`base_url`) across LLM, embeddings, and multimodal adapters

## M4 - Video MVP Path
### Objective
Add video understanding with frame sampling + temporal aggregation and retrieval-aware temporal grounding.

### Outputs
- Code modules:
  - `app/video/frame_sampler.py`
  - `app/video/temporal_aggregator.py`
  - `app/video/adapter.py`
- Endpoints:
  - `/video/analyze`
- Tests:
  - Frame extraction correctness tests.
  - Temporal summary coherence tests.
  - Retrieval-grounded timeline citation tests.
  - Latency budget tests.
- Docs updates:
  - Add explicit text+image vs text+image+video complexity delta.

### Risks and Mitigation
- Risk: High latency and compute cost.
- Mitigation: Configurable sampling rate and early-stop policy for long videos.

### Rollback/Fallback
- Degrade to keyframe-only analysis when full temporal pass exceeds budget.

### Definition of Done
- Video endpoint produces coherent summary with temporal cues.
- Delta analysis section is completed in didactic doc.

### Implementation Status
- Current branch status (2026-03-06): implemented in current working tree and test-covered.
- Delivered now:
  - `VideoFrameSampler` now performs strict local/remote decode-based sampling (`cv2` required), with explicit failure when decode is unavailable.
  - `VideoAnalysisAdapter` now orchestrates per-frame vision analysis plus provider-level video summary.
  - `TemporalAggregator` now composes events from frame-level findings and prioritizes those findings for top-level summary.
  - `/video/analyze` preserves response contract while emitting timeline-evidenced key events (`[t=...s][source=...]`) and budget-aware `processed_frames`.
  - Multimodal ingestion now supports a configurable strategy: default VL-first video indexing (direct `video_url` embedding) with optional frame-analysis enrichment when enabled.

## M5 - Production Hardening
### Objective
Make runtime behavior reliable, observable, and cost-aware.

### Outputs
- Code modules:
  - `app/core/retry.py`
  - `app/core/cache.py`
  - `app/core/model_router.py`
  - `app/core/errors.py`
- Endpoints:
  - `/metrics` finalized.
- Tests:
  - Retry idempotency tests.
  - Cache hit/miss behavior tests.
  - Error taxonomy and mapping tests.
  - Structured-output schema conformance tests for all answering endpoints.
  - Online/offline retrieval quality regression gate tests.
- Docs updates:
  - Resilience and cost optimization explanations.

### Risks and Mitigation
- Risk: Hidden failures due to poor logging correlation.
- Mitigation: Mandatory request/trace IDs and stage-level structured events.

### Rollback/Fallback
- Disable advanced routing and pin to baseline model if quality regression detected.

### Definition of Done
- Logs, retries, cache, and routing are measurable and documented.
- Non-functional acceptance checks are green.

## M5.1 - Context Compaction (Codex-style)
### Objective
Prevent context-window overflow by compacting long agent/query sessions into durable summary checkpoints while preserving key constraints, citations, and unresolved tasks.

### Outputs
- Code modules:
  - `app/core/context_compaction.py`
  - `app/agents/context_manager.py`
  - `app/contracts/context.py` (optional compacted-context schema)
- Integration points:
  - Orchestrator pre-step guard (trigger compaction when token threshold is exceeded).
  - Query pipeline guard for long multi-turn sessions.
- Tests:
  - Trigger-threshold tests.
  - Summary-quality invariants (must preserve goals, constraints, citations, open tasks).
  - No-regression tests for answer quality after compaction.
- Docs updates:
  - Document compaction strategy and safety invariants.

### Risks and Mitigation
- Risk: critical context loss after aggressive pruning.
- Mitigation: enforce pinned-context blocks (requirements, constraints, tool outputs, citations) and validate compaction output schema before replacing history.

### Rollback/Fallback
- Disable automatic compaction and continue with warning-only mode if quality regression is detected.

### Definition of Done
- Session state compacts automatically under budget pressure.
- Compacted state preserves critical facts and constraints in tests.
- Latency/token usage improves on long-session benchmarks without citation-quality regression.

### Implementation Status
- Current branch status (2026-03-08): M5.1 vertical slice implemented in current working tree.
- Delivered in this slice:
  - `ContextCheckpoint` + pinned-context schema (`goals`, `constraints`, `citations`, `open_tasks`, `tool_outputs`).
  - Threshold-based compactor and context manager.
  - Orchestrator pre-step compaction guard with runtime telemetry event (`reason_code=context_compaction`).
  - Regression tests for trigger/invariants and runtime event emission.

## M5.2 - Steering and Policy Controls
### Objective
Add explicit steering controls so users/operators can shape style, risk posture, tool usage, and citation strictness without changing prompt templates manually.

### Outputs
- Code modules:
  - `app/core/steering.py`
  - `app/contracts/steering.py`
  - route wiring updates in `app/api/routes/query.py` and `app/api/routes/agents.py`
- Features:
  - Steering profiles (e.g., `balanced`, `concise`, `strict-grounded`, `creative`).
  - Tool-use steering (allow/deny/require list).
  - Grounding steering (minimum citation requirement / abstention policy).
- Tests:
  - Profile application tests.
  - Tool steering enforcement tests.
  - Safety/grounding regression tests.
- Docs updates:
  - Steering profile definitions and operational guidance.

### Risks and Mitigation
- Risk: conflicting steering constraints degrade answer utility.
- Mitigation: deterministic precedence rules and runtime conflict diagnostics.

### Rollback/Fallback
- Fallback to default `balanced` steering profile if requested profile is invalid or conflicting.

### Definition of Done
- Steering is configurable per request/session and observable in traces.
- Tool and grounding policies are enforced consistently.
- Default behavior remains backward compatible for existing clients.

### Implementation Status
- Current branch status (2026-03-08): M5.2 baseline implemented in current working tree.
- Delivered in this slice:
  - Steering contracts (`SteeringRequest`, `ToolPolicy`, `GroundingPolicy`) and runtime application helpers.
  - Query/agent route integration for profile resolution, answer-style shaping, tool policy filtering, and grounding-abstention enforcement.
  - Regression tests for tool-policy resolution, strict-grounded abstention, creative style, and route-level steering metadata.

## M5.3 - Multimodal Embedding Stack (Qwen3-VL + Multivector DB)
### Objective
Store and retrieve text, image, screenshot, and video evidence in a shared multimodal vector space, then rerank with cross-modal precision.

### Outputs
- Code modules:
  - `app/rag/multimodal_embeddings.py`
  - `app/interfaces/multimodal_embedding.py`
  - `app/storage/qdrant_store.py` (named vectors + optional multivectors path)
  - `app/rag/ingestion.py` (segment-level multimodal vector writes)
  - `app/rag/retriever.py` (multimodal dense first-pass + rerank stage)
  - `app/core/config.py` (provider/model/runtime knobs)
- Retrieval architecture:
  - Embedding stage: `Qwen3-VL-Embedding` (2B or 8B).
  - Reranking stage: `Qwen3-VL-Reranker` (2B or 8B).
  - Vector DB: Qdrant named vectors for modality-aware search.
- Tests:
  - Multimodal ingest/search parity tests (text-image-video).
  - Cross-modal query tests (text->image/video and image->text retrieval).
  - Reranker lift tests on top-k precision.
  - Storage contract tests for named vectors / vector-name routing.
- Docs updates:
  - End-to-end architecture + operation guide for multimodal retrieval at scale.

### Risks and Mitigation
- Risk: ingestion/runtime cost spikes from media-heavy indexing.
- Mitigation: adaptive frame sampling, pixel/frame caps, and staged retrieval budgets.

- Risk: vector-space drift across modalities.
- Mitigation: single embedding family for all modalities + reranker consistency checks.

### Rollback/Fallback
- Fall back to canonical text evidence retrieval when multimodal embedding provider is unavailable.

### Definition of Done
- One query path can retrieve mixed-modal evidence from one collection.
- Reranking consistently improves retrieval relevance on multimodal test sets.
- Existing `/query`, `/vision/analyze`, `/video/analyze` contracts remain backward compatible.

### Implementation Status
- Current branch status (2026-03-09): implemented in current working tree.
- Delivered in this slice:
  - Added multimodal embedding abstraction (`app/interfaces/multimodal_embedding.py`) and provider builder (`app/rag/multimodal_embeddings.py`) with deterministic fallback plus true VL provider wiring (`qwen3_vl`) for image/video-aware embedding payloads.
  - Reworked ingest/retrieval path to write/query dual dense vectors (`text_dense`, `mm_dense`) through named-vector-capable stores while preserving backward-compatible fallback behavior.
  - Added true VL reranker provider path (`qwen3_vl`) for multimodal query-document reranking, plus OpenAI-compatible LLM reranker fallback profile.
  - Updated video ingestion policy to favor direct VL embeddings first; frame/timeline analysis is now optional enrichment (`MMAA_MULTIMODAL_VIDEO_INGEST_ENRICH_WITH_ANALYSIS`).
  - Added optional local video-audio transcription during ingestion (Whisper + ffmpeg) so spoken content is indexed as timestamped retrieval evidence (`MMAA_MULTIMODAL_VIDEO_AUDIO_TRANSCRIPTION_*`).
  - Extended Qdrant adapter with named vector upsert/search routing and collection bootstrap support for text+multimodal vector fields.
  - Added M5.3 regression suites (`tests/test_m53_multimodal_stack.py`, `tests/test_m53_openai_reranker.py`) covering ingest/search parity, cross-modal retrieval, reranker lift, named-vector routing, and OpenAI-compatible reranker behavior.

## M6 - Evaluation and Deployment
### Objective
Operationalize benchmarking and deployment artifacts.

### Outputs
- Code modules:
  - `evaluation/runner.py`
  - `evaluation/retrieval_metrics.py`
  - `evaluation/generation_metrics.py`
  - `evaluation/datasets/*.jsonl`
  - `deployment/Dockerfile`
  - `deployment/docker-compose.yml`
- Endpoints:
  - `/metrics` includes aggregate benchmark results.
- Tests:
  - Reproducible benchmark run.
  - Retrieval benchmark reproducibility test (recall@k, MRR/NDCG).
  - Grounded generation benchmark reproducibility test (faithfulness, citation precision).
  - Container startup and health checks.
- Docs updates:
  - Deployment runbook and metric interpretation guide.

### Risks and Mitigation
- Risk: Benchmarks not reproducible across environments.
- Mitigation: Pin dataset version, config snapshot, and model routing profile in each run.

### Rollback/Fallback
- Keep previous stable image tag and rollback deployment by tag.

### Definition of Done
- Benchmark report includes accuracy, latency, and cost.
- Benchmark report includes retrieval and grounding metrics by milestone profile.
- Service runs from container and exposes public-ready config.

## 4. Cross-Milestone Quality Gates
- API contracts stay backward compatible unless explicitly versioned.
- Each milestone must add or update automated tests.
- Didactic traceability doc must be updated in the same PR as implementation.

## 5. Final Integration and Demo Sequence
1. Start backend service and validate `/health`.
2. Launch Streamlit frontend and review the architecture diagram.
3. Ingest a PDF and one URL with `/ingest/documents`.
4. Run text question through `/query` and inspect citations.
5. Run multi-agent execution through `/agents/run` and inspect trace.
6. Run `/vision/analyze` on sample image.
7. Run `/video/analyze` on sample short video.
8. Fetch `/metrics` and present latency/cost/accuracy summary.

## 6. Suggested Execution Rhythm
- One milestone per branch.
- Merge only when Definition of Done and quality gates are satisfied.
- After merge, immediately append technical decisions and lessons to didactic traceability.
