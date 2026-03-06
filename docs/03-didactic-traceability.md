# 03 - Didactic Traceability

## 1. Purpose
This document explains learning intent and technical reasoning during implementation. It must be updated continuously as milestones are delivered.

## 2. How to Use This Document
- Read the learning map before implementing a milestone.
- During implementation, append technical decisions with timestamped entries.
- After implementation, record observed outcomes, failures, and what changed.

## 3. Learning Map (Feature -> Concepts -> Technologies)
| Milestone | Feature | Core concepts to learn | Main technologies |
| --- | --- | --- | --- |
| M0 | Architecture shell | Interfaces, dependency inversion, schema-first APIs, structured logging | FastAPI, Pydantic, Python typing |
| M1 | Text RAG | Chunking, neural embeddings, hybrid retrieval, reranking, retrieval grounding | PostgreSQL, PGVector, neural embedding adapters, rank fusion |
| M2 | Multi-agent ReAct | Agent roles, shared state, tool routing, bounded reasoning loops, durable checkpoints | LangGraph StateGraph, MCP tool adapters, tool registry patterns |
| M2.2 | Frontend architecture visibility | API contract usability, architecture communication, backend/frontend coupling boundaries | Streamlit, Graphviz DOT, HTTP API clients |
| M3 | Image path | Input preprocessing, multimodal inference, evidence fusion | Vision model adapters |
| M4 | Video path | Frame sampling, temporal aggregation, budget-aware processing | Video pipeline utilities |
| M5 | Hardening | Retry strategy, caching layers, model routing, error taxonomy | Async IO, resilient API patterns |
| M6 | Evaluation + deploy | Metric design, retrieval/generation evaluation, reproducibility, containerization | Benchmark runners, Docker, retrieval/grounding metrics |

## 4. Didactic Cards by Milestone

## M0 - Skeleton and Contracts
### How it works
API endpoints validate requests using Pydantic schemas and delegate execution to interface-driven services.

### Why it works
Separating contracts from implementation reduces coupling and allows replacing providers without changing endpoint behavior.

### Why this technology fits
FastAPI + Pydantic gives strict request/response validation with low ceremony and strong learning feedback.

## M1 - Text RAG
### How it works
Documents are ingested, chunked, embedded, and stored as vectors plus metadata. Queries run hybrid retrieval (dense + lexical), optional reranking, then answer generation from selected evidence.

### Why it works
Dense retrieval captures semantic similarity; lexical retrieval captures exact-term relevance; reranking improves final ordering quality. Their combination increases robustness across query types.

### Why this technology fits
PGVector inside PostgreSQL keeps vectors and metadata in one store, while provider adapters allow neural embedding upgrades without API contract changes.

### Retrieval design (M1 modern baseline)
- Deterministic chunking remains configurable for reproducible tests.
- Embedding layer is provider-pluggable: neural providers in production profile, deterministic fallback for local/test resilience.
- Vector-store layer is adapter-based: pgvector baseline plus optional external vector DB primary with pgvector/in-memory fallback coexistence.
- Retrieval uses dense search and lexical search in parallel, fuses results (RRF), and supports reranker hook for final ordering.
- Ingestion stores vector and trace metadata (`source`, `chunk_id`, `offset`, `snippet`) for grounded citations.

### PGVector rationale (with pragmatic fallback)
- Primary path: PostgreSQL + PGVector + JSONB in a single table to keep vectors and metadata strongly linked.
- Fallback path: in-memory vector index with the same metadata contract to preserve API behavior in local/dev environments without a running DB.
- Result: endpoint contracts stay stable while infra requirements scale from local prototype to production database deployment.

### M2+ Improvement Targets Detected from M1 Review
- M2: durable orchestration with checkpoint/resume and idempotent tool-call replay.
- M2: MCP adapter boundary in tool registry so tool integration is protocol-driven.
- M3/M4: evidence-grounded multimodal outputs with citation-style visual/temporal references.
- M5: structured output validation and automated regression gates for retrieval quality.
- M6: benchmark suite with retrieval metrics (recall@k, MRR/NDCG) plus grounding/faithfulness metrics.

## M2 - ReAct + Multi-agent
### How it works
A shared `AgentState` flows through role-specific agents (`ResearchAgent`, `AnalystAgent`, `AnswerAgent`) coordinated by `AgentOrchestrator`. Tool calls are routed through `ToolRegistry` with timeout/retry guards, and checkpoints are persisted after each stage for resume-safe execution.

### Why it works
Role decomposition narrows each reasoning step, while explicit state snapshots and bounded tool budgets make failures diagnosable and recovery predictable.

### Why this technology fits
An orchestrator + state-machine pattern keeps M2 implementation lightweight while preserving the same control-flow concepts needed for future LangGraph migration.

### M2 implementation notes
- Stage order is explicit and test-covered: `research_agent -> analyst_agent -> answer_agent`.
- Tool calls are bounded by configurable budget, timeout, and retry limits.
- Resume behavior reuses checkpoints and skips already-completed stages to avoid duplicate tool execution.
- MCP adapter boundary is introduced as an integration seam (`mcp_adapter.py`) without forcing transport/runtime coupling yet.
- Runtime orchestration now executes on LangGraph `StateGraph` while preserving `AgentState` and endpoint contracts.
- LLM selection is provider-based (`auto`/`openai`/`heuristic`) and no longer hardwired to a stub class.
- Tool discovery is exposed through `/agents/tools`, and `/agents/run` defaults to all registered tools when `tools` is omitted.

## M2.2 - Streamlit Frontend + Architecture Visualization
### How it works
A Streamlit UI calls existing backend routes directly and renders a Graphviz DOT architecture diagram that maps core components (frontend, API, orchestration, RAG, multimodal routes, observability).

### Why it works
Keeping one tab per route makes API contracts inspectable by humans while preserving the same backend behavior already validated by tests.

### Why this technology fits
Streamlit allows a low-friction teaching UI without introducing a heavy frontend framework during backend milestone progression.

### M2.2 implementation notes
- Frontend scope is intentionally thin: it reuses existing endpoints and adds no new backend logic.
- Architecture description is centralized in `frontend/architecture.py` for reuse and testability.
- A frontend helper test validates diagram content and flow explanation presence.
- Agent tool selection moved from free-text input to discoverable multiselect, populated from `/agents/tools`.
- Ingestion UX now supports clipboard image paste (`Ctrl+V`) in addition to file upload.

## M3 - Image Multimodal
### How it works
Image and video sources can be ingested through the same indexing endpoint and transformed into textual semantic descriptors through multimodal clients before embedding and vector storage.

### Why it works
Multimodal-to-text descriptors allow image/video assets to participate in the same retrieval and citation flow as text documents.

### Why this technology fits
A multimodal adapter layer keeps provider flexibility while minimizing impact on existing vector-store and retrieval contracts.

### External Landscape Scan (2026-03-04)
- Industrial VLM APIs (OpenAI, Anthropic, Google) converge on image+text prompt interfaces and production safety controls.
- Open-source VLM research is moving toward stronger visual reasoning and dynamic resolution support (e.g., Qwen2.5-VL).
- VLA research (RT-2, OpenVLA) reinforces that perception-to-action stacks benefit from explicit grounding interfaces and robust state handling.
- Implementation implication for this project: keep strict adapter boundaries and emit explicit evidence tags in multimodal outputs so future VLA/tool integration can reuse grounded observations.

## M4 - Video Multimodal
### How it works
Video is sampled into frames, frame-level observations are generated, and a temporal aggregator composes a sequence-aware summary.

### Why it works
Temporal aggregation captures event order and progression, which single-frame analysis cannot represent.

### Why this technology fits
A staged video pipeline gives direct control over compute budget and output quality tradeoffs.

## M5 - Hardening
### How it works
Retries handle transient failures, caching avoids repeated expensive calls, and model routing balances cost and quality.

### Why it works
Operational safeguards reduce tail failure rates and cost variance under realistic workloads.

### Why this technology fits
Composable middleware-style utilities keep resilience logic centralized and testable.

## M6 - Evaluation + Deploy
### How it works
A fixed dataset runner executes scenarios and records accuracy, latency, and cost metrics. Container packaging standardizes runtime behavior.

### Why it works
Reproducible evaluation converts subjective quality into trackable engineering signals.

### Why this technology fits
Docker enables consistent environment portability and easier public deployment.


## 5. Explicit Complexity Delta: Text+Image vs Text+Image+Video
| Aspect | Text + Image | Text + Image + Video |
| --- | --- | --- |
| Data volume | Low to medium per request | Medium to high due to frame extraction |
| Latency profile | Single-pass multimodal inference | Multi-stage pipeline with sampling + aggregation |
| Compute cost | Moderate | Higher and more variable |
| Failure modes | Image quality, OCR/vision ambiguity | Frame drift, temporal incoherence, sampling bias |
| Observability needs | Request-level traces | Stage-level traces per frame batch and aggregator |
| Tuning parameters | Prompt and image preprocessing | Sampling rate, frame window, temporal weighting |

## 6. Technical Decision Log (Inline)
Use this entry template for every major decision:

- Date:
- Milestone:
- Context:
- Decision:
- Alternatives considered:
- Why chosen:
- Expected impact:
- Observed outcome:
- Would we choose it again?:
- Affected modules:

### Initial Decisions

- Date: 2026-02-28
- Milestone: M0
- Context: Needed to prioritize implementation speed and learning feedback.
- Decision: CLI-first delivery; postpone web UI.
- Alternatives considered: web-first, parallel CLI+web.
- Why chosen: reduces surface area and keeps focus on backend architecture.
- Expected impact: faster milestone throughput and easier debugging.
- Observed outcome: M0 backend scaffold delivered without frontend overhead; route contracts and tests were completed in one pass.
- Would we choose it again?: yes.
- Affected modules: `app/api/*`, `app/main.py`, `tests/test_api_contracts.py`.

- Date: 2026-02-28
- Milestone: M0
- Context: Provider lock-in would reduce experimentation.
- Decision: enforce provider abstraction supporting hosted and local models.
- Alternatives considered: OpenAI-only adapter, local-only adapter.
- Why chosen: maximizes learning and portability.
- Expected impact: slightly higher upfront complexity, better long-term flexibility.
- Observed outcome: interface contracts plus stub dependency container were added and consumed by route handlers.
- Would we choose it again?: yes.
- Affected modules: `app/interfaces/*`, `app/core/dependencies.py`, `app/api/routes/*`.

- Date: 2026-02-28
- Milestone: M0
- Context: Multimodal scope definition for MVP.
- Decision: include video in MVP and explicitly document added complexity vs image-only path.
- Alternatives considered: defer video to post-MVP.
- Why chosen: stronger learning value and clearer understanding of temporal modeling tradeoffs.
- Expected impact: higher implementation risk, higher educational payoff.
- Observed outcome: M0 includes explicit `/vision/analyze` and `/video/analyze` contracts, preserving the planned implementation path.
- Would we choose it again?: yes.
- Affected modules: `app/api/routes/vision.py`, `app/api/routes/video.py`, `docs/02-implementation-roadmap.md`.

- Date: 2026-02-28
- Milestone: M0
- Context: Domain scope for assistant use cases.
- Decision: exclude crypto-specific integration from all milestones.
- Alternatives considered: include sample crypto tool.
- Why chosen: keeps architecture domain-agnostic and focused on core assistant capabilities.
- Expected impact: cleaner boundaries and reusable platform design.
- Observed outcome: M0 routes, schemas, and tests are domain-neutral and avoid vertical coupling.
- Would we choose it again?: yes.
- Affected modules: all milestone scopes.

- Date: 2026-02-28
- Milestone: M1
- Context: Needed a complete ingest->retrieve path while keeping local development friction low.
- Decision: implement PGVector store with automatic in-memory fallback when PostgreSQL/psycopg is unavailable.
- Alternatives considered: hard-fail without PostgreSQL, pure in-memory only.
- Why chosen: preserves production architecture direction while avoiding local blockers.
- Expected impact: stable API behavior across environments, with clear upgrade path to full DB-backed persistence.
- Observed outcome: M1 tests validate ingestion/query flow and citation grounding without requiring external DB setup.
- Would we choose it again?: yes.
- Affected modules: `app/storage/pgvector_store.py`, `app/core/dependencies.py`, `app/api/routes/ingest.py`, `app/api/routes/query.py`.

- Date: 2026-02-28
- Milestone: M1
- Context: Need predictable retrieval behavior for automated tests and didactic traceability.
- Decision: use deterministic hashing-based embeddings and deterministic chunk splitting.
- Alternatives considered: random vectors for stubs, external embedding provider from day one.
- Why chosen: deterministic behavior makes regressions easy to detect and reason about.
- Expected impact: repeatable retrieval smoke tests and easier tuning of chunk parameters.
- Observed outcome: chunking determinism and retrieval tests are stable across runs.
- Would we choose it again?: yes.
- Affected modules: `app/rag/chunking.py`, `app/rag/embeddings.py`, `app/rag/ingestion.py`, `app/rag/retriever.py`, `tests/test_m1_rag.py`.

- Date: 2026-03-01
- Milestone: M1
- Context: Deterministic hash embeddings provide reproducibility but weak semantic generalization.
- Decision: upgrade M1 baseline to pluggable neural embeddings + hybrid retrieval + reranker hook while keeping deterministic fallback.
- Alternatives considered: keep current hash baseline, move directly to full agentic retrieval in M2.
- Why chosen: improves retrieval quality now without violating milestone boundaries.
- Expected impact: better semantic recall and more stable relevance ranking across mixed query types.
- Observed outcome: provider-pluggable embedding layer, hybrid retrieval with RRF, and reranker hook are implemented with regression tests; M1 suite remains green.
- Would we choose it again?: yes.
- Affected modules: `app/interfaces/embedding.py`, `app/interfaces/reranker.py`, `app/interfaces/vector_store.py`, `app/rag/embeddings.py`, `app/rag/retriever.py`, `app/rag/reranker.py`, `app/storage/pgvector_store.py`, `app/core/config.py`, `app/core/dependencies.py`, `tests/test_m1_rag.py`.

- Date: 2026-03-03
- Milestone: M1
- Context: Need a production-grade external vector DB option while preserving local reliability and existing pgvector behavior.
- Decision: add Qdrant adapter and fallback-wrapper store so external DB can coexist with pgvector/in-memory fallback.
- Alternatives considered: hard switch to external store only, keep pgvector-only path.
- Why chosen: de-risks rollout and keeps development/test workflows stable while enabling production external vector DB usage.
- Expected impact: easier production adoption of managed vector DB while preserving zero-downtime fallback behavior.
- Observed outcome: Qdrant adapter plus fallback store implemented and covered by adapter tests; full suite remains green.
- Would we choose it again?: yes.
- Affected modules: `app/storage/qdrant_store.py`, `app/storage/fallback_vector_store.py`, `app/core/config.py`, `app/core/dependencies.py`, `tests/test_vector_store_fallback.py`.

- Date: 2026-03-04
- Milestone: M2
- Context: Needed to replace placeholder `/agents/run` behavior with observable multi-agent orchestration while keeping deterministic tests.
- Decision: implement an explicit orchestrator with role agents, registry-based tool execution, and checkpoint resume that skips completed stages.
- Alternatives considered: single monolithic agent prompt, direct LangGraph dependency in M2.
- Why chosen: provides clear state transitions and debugging hooks with minimal runtime complexity.
- Expected impact: predictable orchestration behavior, bounded tool execution, and safer continuation after partial runs.
- Observed outcome: M2 tests cover stage transitions, tool selection, timeout/retry, and checkpoint resume idempotency; suite remains green.
- Would we choose it again?: yes.
- Affected modules: `app/agents/*`, `app/tools/registry.py`, `app/tools/mcp_adapter.py`, `app/api/routes/agents.py`, `app/core/dependencies.py`, `app/core/config.py`, `tests/test_m2_agents.py`, `tests/test_m2_checkpoint_resume.py`.

- Date: 2026-03-04
- Milestone: M2.2
- Context: Backend milestones became harder to explain quickly to new reviewers without a visual system map and interactive route surface.
- Decision: add a Streamlit frontend with architecture diagram + route playground instead of waiting for a full product UI milestone.
- Alternatives considered: keep API-only flow, build a React frontend now.
- Why chosen: fastest path to improve explainability and learning feedback with minimal coupling risk.
- Expected impact: easier onboarding/review and clearer communication of module boundaries.
- Observed outcome: architecture can be inspected visually and core backend routes are callable from one UI.
- Would we choose it again?: yes.
- Affected modules: `frontend/streamlit_app.py`, `frontend/architecture.py`, `tests/test_frontend_architecture.py`, `requirements.txt`, `pyproject.toml`.

- Date: 2026-03-04
- Milestone: M2
- Context: Orchestration logic was implemented as a custom linear runner, while milestone intent targeted LangGraph-ready orchestration semantics.
- Decision: migrate orchestrator runtime to LangGraph `StateGraph` and keep `AgentState` as shared payload object.
- Alternatives considered: keep custom pipeline, rewrite all agents around LangChain abstractions.
- Why chosen: delivers actual graph runtime now without destabilizing current tests/contracts.
- Expected impact: cleaner evolution path for branching/conditional agent flows and better parity with production orchestration patterns.
- Observed outcome: stage sequencing and checkpoint-resume behavior remained stable and test-covered after migration.
- Would we choose it again?: yes.
- Affected modules: `app/agents/orchestrator.py`, `tests/test_m2_agents.py`, `tests/test_m2_checkpoint_resume.py`, `requirements.txt`, `pyproject.toml`.

- Date: 2026-03-04
- Milestone: M2-M3 bridge
- Context: Ingestion path handled text/PDF/URL but excluded image/video assets from the shared retrieval index.
- Decision: extend `DocumentIngestionService` to ingest image/video sources through multimodal clients and persist modality metadata in the same vector pipeline.
- Alternatives considered: separate multimodal index endpoint, postpone ingestion until full M3 endpoint redesign.
- Why chosen: enables immediate multimodal retrieval experimentation while preserving one ingestion contract.
- Expected impact: image/video assets become retrievable evidence in `/query` and agent workflows.
- Observed outcome: modality-tagged chunks are indexed for image/video sources and validated by automated tests.
- Would we choose it again?: yes.
- Affected modules: `app/rag/ingestion.py`, `app/multimodal/clients.py`, `app/core/dependencies.py`, `tests/test_llm_and_multimodal_ingestion.py`.

- Date: 2026-03-04
- Milestone: M3
- Context: `/vision/analyze` still relied on direct client output shaping, missing explicit preprocessing and evidence-fusion modules requested by roadmap.
- Decision: implement dedicated `VisionPreprocessor`, `VisionAdapter`, and `VisionFusion` modules and route `/vision/analyze` through this pipeline.
- Alternatives considered: keep inline route logic, defer evidence fusion to M4.
- Why chosen: aligns with roadmap design boundaries and provides explicit grounding now.
- Expected impact: clearer separation of concerns, better validation, and auditable evidence references in findings.
- Observed outcome: endpoint now validates type/size, returns evidence-tagged findings, and is covered by dedicated M3 tests.
- Would we choose it again?: yes.
- Affected modules: `app/vision/preprocess.py`, `app/vision/adapter.py`, `app/vision/fusion.py`, `app/api/routes/vision.py`, `tests/test_m3_vision.py`.

- Date: 2026-03-04
- Milestone: M3-M5 bridge
- Context: Need to run fully local/self-hosted model stacks (containerized) without changing API route contracts.
- Decision: add OpenAI-compatible `base_url` support across LLM, embedding, and multimodal adapters; accept `base_url` as alternative to API key; normalize local image file inputs to data URLs for VLM compatibility.
- Alternatives considered: add provider-specific adapters first (`ollama`, `vllm`) and postpone shared endpoint support.
- Why chosen: keeps provider surface minimal while enabling multiple local backends immediately through one contract.
- Expected impact: easier local deployment on consumer GPUs and smoother switch between hosted and local providers.
- Observed outcome: env-driven local endpoint routing now works for all three inference surfaces with automated tests.
- Would we choose it again?: yes.
- Affected modules: `app/llm/clients.py`, `app/rag/embeddings.py`, `app/multimodal/clients.py`, `app/core/config.py`, `app/core/dependencies.py`, `tests/test_llm_and_multimodal_ingestion.py`.

### What Changed in Understanding (M2)
- Durable orchestration value appears earlier than expected: checkpointing is useful even before distributed execution.
- Tool reliability controls (timeout + retry + budget) are first-class orchestration concerns, not auxiliary utilities.
- A protocol adapter boundary (MCP) can be added incrementally without blocking core orchestration delivery.
- Migrating to LangGraph did not require rewriting agent internals when state boundaries were already explicit.

### What Failed and How It Was Fixed (M2)
- Failure: early orchestrator draft could rebuild from checkpoint but still risk rerunning previously completed stages.
- Fix: resume path now reconstructs `AgentState` from snapshot and skips stages already present in `steps`, preventing repeated tool calls.
- Verification: `test_checkpoint_resume_avoids_repeating_completed_tool_calls` confirms no duplicate tool execution during resume.

### What Changed in Understanding (M3 bridge)
- Multimodal ingestion can be introduced incrementally by converting visual/video inputs into grounded descriptors before embedding.
- A single vector index with modality metadata is a practical bridge before full cross-modal ranking models are added.

### What Changed in Understanding (M3)
- Dedicated preprocessing/fusion stages are necessary even with strong VLMs, because production behavior depends on deterministic validation and traceable evidence formatting.
- A user-facing multimodal endpoint should preserve raw model summary and separately expose structured findings with evidence metadata.
- Webpage URLs frequently point to HTML instead of direct images; preprocessing should resolve concrete image assets before VLM invocation.

### What Failed and How It Was Fixed (M3 bridge)
- Failure: initial ingestion path attempted to read every source as text bytes, which does not generalize to image/video URIs.
- Fix: ingestion now resolves modality first, then routes image/video to multimodal analysis clients and text sources to byte/text loaders.
- Verification: `test_image_and_video_are_ingested_with_modality_metadata` confirms modality-tagged indexing for both media types.

### What Failed and How It Was Fixed (M3)
- Failure: vision request validation misclassified Windows local paths as unsupported URI schemes.
- Fix: preprocessor now detects drive-letter paths and treats them as local files.
- Verification: `test_vision_preprocessor_extracts_deterministic_fixture_metadata` and route-level grounding tests are green.

- Failure: users often passed article URLs (HTML pages) to `/vision/analyze`, causing provider failures or degraded metadata-only responses.
- Fix: preprocessor now resolves webpage image candidates (`og:image`, `twitter:image`, and image URLs in markup), downloads the resolved image, and forwards a concrete data URI to the vision model.
- Verification: dedicated tests for webpage-resolution success/failure are green and real URL checks now resolve image payloads.

- Date: 2026-03-05
- Milestone: M2-M2.2 bridge
- Context: users could not discover available tool names and had to type internal IDs manually.
- Decision: add `GET /agents/tools` plus frontend multiselect populated from tool catalog.
- Alternatives considered: keep free-text tools input, hardcode labels in frontend.
- Why chosen: keeps UX simple while preserving backend tool extensibility.
- Expected impact: fewer invalid tool names and clearer user-facing control.
- Observed outcome: route contract and frontend now show tool names/descriptions directly.
- Would we choose it again?: yes.
- Affected modules: `app/api/routes/agents.py`, `app/tools/registry.py`, `app/contracts/schemas.py`, `frontend/streamlit_app.py`, `tests/test_api_contracts.py`.

- Date: 2026-03-05
- Milestone: M2.2
- Context: many desktop workflows involve screenshot clipboard usage rather than saved files.
- Decision: support clipboard image paste in ingestion flows using `streamlit-paste-button`.
- Alternatives considered: upload-only flow, custom component from scratch.
- Why chosen: fastest reliable UX improvement with low maintenance overhead.
- Expected impact: quicker image ingestion during demos and manual testing.
- Observed outcome: pasted images are persisted to temp and included in ingest source list.
- Would we choose it again?: yes.
- Affected modules: `frontend/streamlit_app.py`, `requirements.txt`, `README.md`.

- Date: 2026-03-05
- Milestone: M3
- Context: `/vision/analyze` input commonly arrives as webpage URLs, not direct image URLs.
- Decision: extend `VisionPreprocessor` to resolve image assets from webpage metadata/markup before inference.
- Alternatives considered: require direct image URLs only, keep degraded-mode fallback.
- Why chosen: improves robustness without changing endpoint contract.
- Expected impact: fewer degraded responses and better real-world success rate.
- Observed outcome: webpage URLs now resolve to concrete image data URIs for model calls.
- Would we choose it again?: yes.
- Affected modules: `app/vision/preprocess.py`, `tests/test_m3_vision.py`, `frontend/architecture.py`.

### What Changed in Understanding (M2.2)
- A lightweight frontend can be introduced without violating milestone isolation if it only consumes stable backend contracts.
- Architecture communication deserves explicit code artifacts (`architecture.py`), not only markdown prose.

### What Failed and How It Was Fixed (M2.2)
- Failure: route-level understanding was fragmented across files and difficult to present in one narrative.
- Fix: added a single architecture diagram plus route tabs in Streamlit that mirrors real backend entrypoints.
- Verification: `test_frontend_architecture.py` checks key diagram elements and explanatory flow points.

## 7. Common Failure Modes and Debugging Heuristics

## RAG
- Failure mode: low-relevance retrieval.
- Heuristic: inspect chunk size/overlap, embedding provider/model, dense/lexical fusion balance, and rerank candidate budget.

- Failure mode: answer without grounding.
- Heuristic: enforce citation-required response schema and reject uncited outputs.

## Agent Orchestration
- Failure mode: tool-call loops.
- Heuristic: set max iteration count, tool budget, and loop-detection guard in state.

- Failure mode: inconsistent agent handoff.
- Heuristic: validate mandatory state keys between agent transitions.

## Multimodal
- Failure mode: incorrect image interpretation.
- Heuristic: add confidence threshold and expose uncertain observations as tentative.

- Failure mode: incoherent video summary.
- Heuristic: lower sampling stride for fast scenes and compare against keyframe-only baseline.

## Operations
- Failure mode: intermittent provider failures.
- Heuristic: apply retry with jitter and fallback model route.

- Failure mode: cost spikes.
- Heuristic: check cache hit rate, token budgets, and model-routing decisions by request class.

## 8. Update Protocol (Mandatory)
For each milestone completion, update this file with:
- At least one new decision log entry.
- One section describing what changed in understanding.
- One section describing what failed and how it was fixed.
- A reference to affected code modules and tests.

## 9. File-Level Traceability Location
File-by-file traceability has been moved to:
- `docs/04-file-traceability-by-milestone.md`

This keeps this document focused on learning intent, milestone rationale, and technical decision history.
