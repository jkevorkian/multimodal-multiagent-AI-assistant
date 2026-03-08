# 04 - File Traceability by Milestone

## 1. Purpose
Keep file-level architecture traceability in one place, separated by milestone, so design intent and implementation behavior are easy to audit over time.

## 2. How to Use This Document
- Navigate by milestone (`M0`, `M1`, ...).
- Each file entry includes:
  - milestone metadata (`introduced`, `updated`, `status`)
  - theoretical role
  - technical/practical role
- `implemented` means the file exists in the repository.
- `planned` means the file is defined in roadmap scope but not implemented yet.

## 3. M0 - Skeleton and Contracts

### 3.1 Project Setup and Docs
`README.md`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: entry-point artifact for project purpose and usage orientation.
- Technical/practical role: documents startup flow and repository context.

`requirements.txt`
- Milestone metadata: introduced `M0`; updated `M2.2`; status `implemented`.
- Theoretical role: minimal dependency manifest for quick bootstrap.
- Technical/practical role: provides base package list for backend, LangGraph orchestration, and Streamlit frontend runtime (including clipboard paste support).

`pyproject.toml`
- Milestone metadata: introduced `M0`; updated `M3`; status `implemented`.
- Theoretical role: canonical packaging and tooling configuration.
- Technical/practical role: declares runtime/dev dependencies and pytest defaults, including LangGraph and provider clients.

`docs/01-requirements.md`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: source of truth for system scope and acceptance criteria.
- Technical/practical role: defines FR/NFR testable requirements and boundaries.

`docs/02-implementation-roadmap.md`
- Milestone metadata: introduced `M0`; updated `M5.2`; status `implemented`.
- Theoretical role: milestone planning artifact for incremental delivery.
- Technical/practical role: defines objectives, outputs, risks, and DoD per milestone.

`docs/03-didactic-traceability.md`
- Milestone metadata: introduced `M0`; updated `M5.2`; status `implemented`.
- Theoretical role: learning/decision narrative over implementation lifecycle.
- Technical/practical role: stores didactic cards, decision log, and troubleshooting heuristics.

`docs/07-local-model-backends.md`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: deployment/options guidance for local and self-hosted model backends.
- Technical/practical role: documents OpenAI-compatible local endpoint profiles (Ollama/vLLM), container setup, and `.env` wiring.

`docs/08-multimodal-research-and-m41.md`
- Milestone metadata: introduced `M4.1`; updated `M4.1`; status `implemented`.
- Theoretical role: research-backed architecture rationale for multimodal retrieval and video analysis upgrades.
- Technical/practical role: records current papers/industry practices and maps them to concrete M4.1 implementation decisions.

### 3.2 Application Composition Root
`app/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: package boundary marker for the application domain.
- Technical/practical role: enables consistent imports under `app.*`.

`app/main.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: composition root for runtime wiring.
- Technical/practical role: creates FastAPI app, configures middleware, and mounts routers.

### 3.3 Core Infrastructure
`app/core/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: namespace for cross-cutting infrastructure concerns.
- Technical/practical role: groups config/logging/dependency modules under `app.core`.

`app/core/config.py`
- Milestone metadata: introduced `M0`; updated `M4`; status `implemented`.
- Theoretical role: centralized environment-driven configuration contract.
- Technical/practical role: defines `Settings`, including app, RAG, agent, LLM-provider, and multimodal tuning parameters.

`app/core/logging.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: observability boundary for request lifecycle tracing.
- Technical/practical role: configures JSON logging and injects request/trace IDs.

`app/core/dependencies.py`
- Milestone metadata: introduced `M0`; updated `M4.1`; status `implemented`.
- Theoretical role: dependency inversion entrypoint between API and services.
- Technical/practical role: builds and provides `ServiceContainer` for endpoint dependencies, including LLM and multimodal client selection.

### 3.4 API Contracts
`app/contracts/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: schema package boundary marker.
- Technical/practical role: keeps request/response contract modules isolated.

`app/contracts/schemas.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role: schema-first API contract.
- Technical/practical role: defines request/response models and validation constraints, including agent-tool catalog response types.

### 3.5 Provider Abstraction Interfaces
`app/interfaces/__init__.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: unified export surface for interface protocols.
- Technical/practical role: re-exports interface types for stable imports, including reranker abstractions.

`app/interfaces/llm.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: abstraction for text generation behavior.
- Technical/practical role: declares `LLMClient.generate`.

`app/interfaces/embedding.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: abstraction for vectorization behavior.
- Technical/practical role: declares `EmbeddingClient.embed_text` and batched `embed_texts`.

`app/interfaces/retriever.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: abstraction for retrieval strategy behavior.
- Technical/practical role: declares `Retriever.retrieve`.

`app/interfaces/vector_store.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: persistence boundary for vector search storage.
- Technical/practical role: declares `upsert`, dense `search`, and lexical `keyword_search`.

`app/interfaces/reranker.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: ranking abstraction between retrieval and generation.
- Technical/practical role: declares async reranking contract over retrieved candidates.

`app/interfaces/vision.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: abstraction for image-analysis behavior.
- Technical/practical role: declares `VisionClient.analyze_image`.

`app/interfaces/video.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: abstraction for video-analysis behavior.
- Technical/practical role: declares `VideoClient.analyze_video` with sampling controls.

`app/interfaces/tool.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: abstraction for pluggable tool execution.
- Technical/practical role: declares tool name and async run signature.

### 3.6 API Layer
`app/api/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: transport layer package marker.
- Technical/practical role: contains router assembly modules.

`app/api/router.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: deterministic API route composition boundary.
- Technical/practical role: includes and mounts all route modules.

`app/api/routes/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: route namespace marker.
- Technical/practical role: enables route imports from a single package path.

`app/api/routes/health.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: liveness/readiness contract boundary.
- Technical/practical role: implements `GET /health`.

`app/api/routes/ingest.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: ingestion boundary from raw sources to indexing pipeline.
- Technical/practical role: implements `POST /ingest/documents`.

`app/api/routes/query.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: query boundary for retrieval-grounded answer generation.
- Technical/practical role: implements `POST /query`.

`app/api/routes/agents.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role: multi-agent orchestration entrypoint contract.
- Technical/practical role: implements `POST /agents/run` using orchestrator-driven role pipeline and `GET /agents/tools` for tool discovery.

`app/api/routes/vision.py`
- Milestone metadata: introduced `M0`; updated `M3`; status `implemented`.
- Theoretical role: image multimodal entrypoint contract.
- Technical/practical role: implements `POST /vision/analyze` using active vision client output with structured finding extraction.

`app/api/routes/video.py`
- Milestone metadata: introduced `M0`; updated `M4.1`; status `implemented`.
- Theoretical role: video multimodal entrypoint contract.
- Technical/practical role: implements `POST /video/analyze` using adapter-backed timeline sampling/aggregation with structured key-event extraction.

`app/api/routes/metrics.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: operational metrics contract boundary.
- Technical/practical role: implements `GET /metrics`.

### 3.7 Test Layer
`tests/test_api_contracts.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role: contract-level regression guard.
- Technical/practical role: validates route presence (including `/agents/tools`), response schema shape, and trace headers.

## 4. M1 - Text RAG Baseline

### 4.1 New M1 Modules
`app/rag/__init__.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: package boundary for RAG pipeline components.
- Technical/practical role: re-exports chunking, embedding selection, reranker selection, ingestion, and retriever modules.

`app/rag/chunking.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: deterministic chunk segmentation strategy.
- Technical/practical role: defines chunk dataclass and overlap-based splitting logic.

`app/rag/embeddings.py`
- Milestone metadata: introduced `M1`; updated `M3`; status `implemented`.
- Theoretical role: embedding engine for semantic similarity.
- Technical/practical role: implements pluggable embedding adapters (deterministic, OpenAI, SentenceTransformers), batch embedding support, OpenAI-compatible base-url routing, and cosine similarity.

`app/rag/ingestion.py`
- Milestone metadata: introduced `M1`; updated `M4.1`; status `implemented`.
- Theoretical role: ingestion orchestration boundary for document indexing.
- Technical/practical role: loads text/image/video sources, routes multimodal analysis when needed, chunks descriptors, embeds, and upserts modality-aware metadata.

`app/rag/retriever.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: retrieval strategy boundary for query-time context selection.
- Technical/practical role: performs hybrid dense+lexical retrieval, reciprocal-rank fusion, reranker hook execution, and output normalization.

`app/rag/reranker.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: relevance refinement layer after candidate retrieval.
- Technical/practical role: provides no-op, lexical overlap, and optional cross-encoder reranking strategies with provider factory.

`app/storage/__init__.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: storage adapter package boundary.
- Technical/practical role: exports PGVector store implementation.

`app/storage/pgvector_store.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: vector persistence adapter for semantic retrieval.
- Technical/practical role: supports PostgreSQL/PGVector path with in-memory fallback behavior.

`app/storage/qdrant_store.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: external vector database adapter for production-grade vector indexing/search.
- Technical/practical role: connects to Qdrant collections for dense vector operations and local lexical metadata search.

`app/storage/fallback_vector_store.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: resilience layer for multi-backend coexistence.
- Technical/practical role: routes reads/writes across primary external vector store and secondary pgvector/in-memory fallback.

`tests/test_m1_rag.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: milestone-specific regression suite for RAG behavior.
- Technical/practical role: tests ingestion flows, chunk determinism, hybrid retrieval behavior, and reranker hook behavior.

`docs/04-file-traceability-by-milestone.md`
- Milestone metadata: introduced `M1`; updated `M3`; status `implemented`.
- Theoretical role: dedicated file-level traceability registry.
- Technical/practical role: separates file mapping concerns from didactic decision narratives.

### 4.2 Existing Files Updated in M1
`app/core/config.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: expanded from app-level settings to RAG system tuning surface.
- Technical/practical change: added embedding provider/model settings, retrieval fusion budgets, reranker settings, and ingestion/vector store tuning knobs.

`app/core/dependencies.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: moved from stub-only wiring toward milestone-aligned service composition.
- Technical/practical change: wires embedding-provider factory, reranker factory, hybrid retriever, and vector-store backend selection/fallback into the container.

`app/storage/pgvector_store.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role change: expanded from vector-only adapter to hybrid retrieval storage component.
- Technical/practical change: adds BM25-like lexical scoring in memory and PostgreSQL full-text ranking path.

`app/contracts/schemas.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role change: extends API contract to expose indexing result granularity.
- Technical/practical change: adds `indexed_chunks` to ingest response and agent-tool catalog models.

`app/api/routes/ingest.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: transitions from placeholder contract to executable ingestion boundary.
- Technical/practical change: calls ingestion service and returns indexed chunk totals.

`app/api/routes/query.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: transitions from stub retrieval flow to grounded citation-producing flow.
- Technical/practical change: executes retrieval over indexed chunks and returns chunk-level citations.

`tests/test_api_contracts.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role change: contract checks now cover M1 ingest output plus M2 agent catalog behavior.
- Technical/practical change: ingest test validates non-zero indexed chunk count; agent tests validate `/agents/tools` and default tool-selection behavior.

`docs/03-didactic-traceability.md`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: remains decision/learning artifact; file-level mapping moved out.
- Technical/practical change: references this document as the source for per-file traceability.

## 5. M2 - ReAct Tools + Multi-Agent Orchestration

`app/agents/__init__.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: package boundary for agent orchestration components.
- Technical/practical role: re-exports state, role agents, checkpoint store, and orchestrator.

`app/agents/state.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: shared state model for multi-agent coordination.
- Technical/practical role: tracks query, context, tool outputs, confidence, trace, and supports snapshot/restore for checkpoint resume.

`app/agents/research_agent.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: evidence-gathering agent role.
- Technical/practical role: runs retrieval, selects eligible tools, and executes tool calls under timeout/retry/budget controls.

`app/agents/analyst_agent.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: intermediate synthesis role.
- Technical/practical role: converts retrieval/tool artifacts into analysis notes and confidence estimate.

`app/agents/answer_agent.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: final composition role.
- Technical/practical role: invokes LLM with staged context to produce final answer and normalized confidence.

`app/agents/orchestrator.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: control-flow coordinator for role sequencing and guardrails.
- Technical/practical role: executes role pipeline on LangGraph `StateGraph`, enforces max-step boundaries, saves checkpoints, and supports resume without repeating completed stages.

`app/llm/__init__.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: package boundary for LLM-provider selection utilities.
- Technical/practical role: exports LLM builder contract for container wiring.

`app/llm/clients.py`
- Milestone metadata: introduced `M2`; updated `M3`; status `implemented`.
- Theoretical role: provider abstraction realization for answer synthesis.
- Technical/practical role: selects OpenAI or heuristic grounded generation client based on runtime settings, including OpenAI-compatible local endpoint routing via base URL.

`app/agents/checkpoint_store.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: durability boundary for long-running agent workflows.
- Technical/practical role: provides null and in-memory checkpoint stores for stage snapshots and resume.

`app/tools/__init__.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: tools package export boundary.
- Technical/practical role: re-exports tool registry and MCP adapter.

`app/tools/registry.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: tool registration and execution boundary.
- Technical/practical role: resolves tool names, exposes discoverable tool descriptions, and executes calls with timeout/retry behavior.

`app/tools/mcp_adapter.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: protocol bridge between internal tool system and MCP-compliant servers.
- Technical/practical role: defines MCP adapter seam and tool proxy contract for later transport integration.

`tests/test_m2_agents.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: behavior regression suite for orchestration mechanics.
- Technical/practical role: validates stage transitions, tool selection scenario, and timeout/retry behavior.

`tests/test_m2_checkpoint_resume.py`
- Milestone metadata: introduced `M2`; updated `M2`; status `implemented`.
- Theoretical role: resilience regression suite for checkpoint recovery.
- Technical/practical role: validates resume flow avoids repeated completed tool calls.

`app/api/routes/agents.py`
- Milestone metadata: introduced `M0`; updated `M2`; status `implemented`.
- Theoretical role change: transitions from placeholder endpoint to true orchestration entrypoint.
- Technical/practical change: delegates `/agents/run` to orchestrator with configurable tool budget and step limits, and adds `/agents/tools` discovery endpoint.

`app/core/dependencies.py`
- Milestone metadata: introduced `M0`; updated `M4.1`; status `implemented`.
- Theoretical role change: expands service composition beyond orchestration into LLM/multimodal provider selection.
- Technical/practical change: wires tool registry, role agents, checkpoint store, LangGraph orchestrator runtime, provider-based LLM selection, and multimodal clients.

`app/core/config.py`
- Milestone metadata: introduced `M0`; updated `M4`; status `implemented`.
- Theoretical role change: now includes orchestration controls in runtime policy surface.
- Technical/practical change: adds M2 settings for orchestration plus runtime knobs for LLM/multimodal providers.

## 6. M2.2 - Streamlit Frontend + Architecture Visualization

`frontend/__init__.py`
- Milestone metadata: introduced `M2.2`; updated `M2.2`; status `implemented`.
- Theoretical role: package boundary marker for frontend artifacts.
- Technical/practical role: enables importable frontend architecture helpers and app modules.

`frontend/architecture.py`
- Milestone metadata: introduced `M2.2`; updated `M5.2`; status `implemented`.
- Theoretical role: architecture communication artifact.
- Technical/practical role: provides Graphviz DOT graph and high-level flow bullets, including LangGraph, tool discovery endpoint, M4.1 video frame-evidence pipeline, and planned runtime controls (context compaction + steering).

`frontend/streamlit_app.py`
- Milestone metadata: introduced `M2.2`; updated `M3`; status `implemented`.
- Theoretical role: human-facing interaction boundary for backend workflows.
- Technical/practical role: renders architecture diagram, adds user-friendly implementation flow tab, preserves route-level playground forms, discovers tools through `/agents/tools`, and supports clipboard image paste ingestion.

`tests/test_frontend_architecture.py`
- Milestone metadata: introduced `M2.2`; updated `M5.2`; status `implemented`.
- Theoretical role: regression guard for architecture communication outputs.
- Technical/practical role: verifies key DOT nodes (including multimodal and LLM layers) and explanatory flow points exist.

## 6.1 M2.3 - Live Runtime Telemetry + Revision Loops

`app/contracts/runtime_events.py`
- Milestone metadata: introduced `M2.3`; status `implemented`.
- Theoretical role: runtime progress contract boundary.
- Technical/practical role: defines stream/snapshot event schemas (`run.started`, `agent.step.*`, `tool.call.*`, `run.completed`, `run.guardrail_triggered`).

`app/core/event_bus.py`
- Milestone metadata: introduced `M2.3`; status `implemented`.
- Theoretical role: event propagation boundary.
- Technical/practical role: publishes ordered run events to SSE consumers and status snapshots.

`app/agents/loop_controller.py`
- Milestone metadata: introduced `M2.3`; status `implemented`.
- Theoretical role: bounded revision control boundary.
- Technical/practical role: enforces deterministic loop exits (`max_steps`, tool budget, stagnation detection, timeout).

`frontend/architecture.py`
- Milestone metadata: introduced `M2.2`; updated `M2.3`; status `implemented`.
- Theoretical role: communicates advanced agent control flow and runtime telemetry model.
- Technical/practical role: provides loop/revision graph and event/guardrail schema helpers for UI rendering.

`frontend/streamlit_app.py`
- Milestone metadata: introduced `M2.2`; updated `M2.3`; status `implemented`.
- Theoretical role: runtime visibility surface for operators/users.
- Technical/practical role: renders loop graph, live-status event taxonomy, guardrail matrix, and runtime status/events panels backed by run endpoints.

`app/api/routes/runs.py`
- Milestone metadata: introduced `M2.3`; status `implemented`.
- Theoretical role: runtime observability API boundary.
- Technical/practical role: exposes `GET /runs/{run_id}/events` (SSE replay/follow) and `GET /runs/{run_id}/status` snapshot retrieval.

`tests/test_m23_runtime_events.py`
- Milestone metadata: introduced `M2.3`; status `implemented`.
- Theoretical role: regression guard for runtime telemetry semantics.
- Technical/practical role: validates event ordering, replay behavior, revision event emission, and guardrail-exit status snapshots.

`tests/test_frontend_architecture.py`
- Milestone metadata: introduced `M2.2`; updated `M2.3`; status `implemented`.
- Theoretical role: regression guard for M2.3 architecture communication helpers.
- Technical/practical role: validates loop graph guardrail nodes plus live-status and loop-guard schema helpers.

## 7. M3 - Image Multimodal Path

`app/vision/__init__.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: package boundary for vision pipeline modules.
- Technical/practical role: exports preprocess/adapter/fusion components for route and service wiring.

`app/multimodal/__init__.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: package boundary for multimodal client adapters.
- Technical/practical role: exports multimodal client builder and selection types.

`app/multimodal/clients.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: adapter layer for image/video analysis providers.
- Technical/practical role: provides OpenAI-backed vision analysis with heuristic fallbacks, OpenAI-compatible base-url routing, and local-file image serialization to data URLs.

`tests/test_llm_and_multimodal_ingestion.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: regression suite for provider fallback and multimodal ingestion behavior.
- Technical/practical role: validates LLM auto fallback, OpenAI-compatible local endpoint wiring, local-file vision payload serialization, and image/video ingestion indexing with modality metadata.

`deployment/docker-compose.ollama.yml`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: local inference infrastructure artifact.
- Technical/practical role: defines base CPU-compatible Ollama topology; GPU enablement is layered via `deployment/docker-compose.ollama.gpu.yml`.

`deployment/docker-compose.ollama.gpu.yml`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: hardware acceleration override artifact.
- Technical/practical role: augments the base Ollama compose profile with GPU device reservations.

`app/vision/preprocess.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: preprocessing boundary for image quality normalization.
- Technical/practical role: validates URI/type/size, resolves webpage URLs to concrete image assets when possible, normalizes local/remote inputs, and emits deterministic image metadata.

`app/vision/adapter.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: provider abstraction realization for vision inference.
- Technical/practical role: orchestrates preprocess + provider call and returns structured analysis artifact.

`app/vision/fusion.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: multimodal fusion boundary between visual and textual evidence.
- Technical/practical role: builds evidence-tagged findings and confidence from analysis output.

`tests/test_m3_vision.py`
- Milestone metadata: introduced `M3`; updated `M3`; status `implemented`.
- Theoretical role: multimodal endpoint quality and validation regression suite.
- Technical/practical role: verifies deterministic fixture handling, type/size validation, and evidence-grounded `/vision/analyze` output.

## 8. M4 - Video MVP Path

`app/video/frame_sampler.py`
- Milestone metadata: introduced `M4`; updated `M4.1`; status `implemented`.
- Theoretical role: temporal sampling strategy boundary.
- Technical/practical role: samples timeline with optional local decode-based frame extraction, frame payload emission, and deterministic fallback.

`app/video/temporal_aggregator.py`
- Milestone metadata: introduced `M4`; updated `M4.1`; status `implemented`.
- Theoretical role: temporal reasoning and synthesis boundary.
- Technical/practical role: composes timeline-oriented key events from frame-level findings or summary fallback with timestamp/source evidence tags.

`app/video/adapter.py`
- Milestone metadata: introduced `M4`; updated `M4.1`; status `implemented`.
- Theoretical role: provider abstraction realization for video inference.
- Technical/practical role: coordinates frame sampling, per-frame vision analysis, provider summary, and temporal aggregation into a route-ready analysis result.

`app/api/routes/video.py`
- Milestone metadata: introduced `M0`; updated `M4.1`; status `implemented`.
- Theoretical role: video multimodal entrypoint contract.
- Technical/practical role: routes requests through `VideoAnalysisAdapter` and returns timeline-evidenced key events plus budget-aware `processed_frames`.

`app/core/config.py`
- Milestone metadata: introduced `M0`; updated `M4`; status `implemented`.
- Theoretical role: centralized environment-driven runtime policy.
- Technical/practical role: now includes video sampling latency budget and timeline aggregation controls (`MMAA_MULTIMODAL_VIDEO_*` tuning fields).

`tests/test_m4_video.py`
- Milestone metadata: introduced `M4`; updated `M4.1`; status `implemented`.
- Theoretical role: regression suite for M4 temporal pipeline behavior.
- Technical/practical role: validates decoded-frame path, frame-level finding integration, latency-budget caps, temporal ordering coherence, adapter composition, and route-level output semantics.

## 9. M5 - Production Hardening (Planned)

`app/core/retry.py`
- Milestone metadata: introduced `M5`; status `planned`.
- Theoretical role: resilience policy boundary for transient failure handling.
- Technical/practical role: implements bounded retries with backoff and jitter.

`app/core/cache.py`
- Milestone metadata: introduced `M5`; status `planned`.
- Theoretical role: performance/cost optimization boundary.
- Technical/practical role: provides caching primitives for embeddings and model responses.

`app/core/model_router.py`
- Milestone metadata: introduced `M5`; status `planned`.
- Theoretical role: quality-cost decision boundary for model selection.
- Technical/practical role: chooses model profiles by request class and fallback conditions.

`app/core/errors.py`
- Milestone metadata: introduced `M5`; status `planned`.
- Theoretical role: error taxonomy boundary across application layers.
- Technical/practical role: standardizes exception types and API error mapping.

## 10. M5.1 - Context Compaction (Planned)

`app/core/context_compaction.py`
- Milestone metadata: introduced `M5.1`; status `planned`.
- Theoretical role: context-budget control boundary for long sessions.
- Technical/practical role: triggers threshold-based compaction and produces durable summary checkpoints with preservation invariants.

`app/agents/context_manager.py`
- Milestone metadata: introduced `M5.1`; status `planned`.
- Theoretical role: orchestration memory lifecycle boundary.
- Technical/practical role: applies compaction decisions to agent state before/after stages.

`app/contracts/context.py`
- Milestone metadata: introduced `M5.1`; status `planned`.
- Theoretical role: compacted-context contract boundary.
- Technical/practical role: defines summary checkpoint schema and pinned-context fields.

## 11. M5.2 - Steering and Policy Controls (Planned)

`app/core/steering.py`
- Milestone metadata: introduced `M5.2`; status `planned`.
- Theoretical role: policy-control boundary for response/tool behavior.
- Technical/practical role: applies steering profiles for style, grounding, and tool policy.

`app/contracts/steering.py`
- Milestone metadata: introduced `M5.2`; status `planned`.
- Theoretical role: steering request/trace schema boundary.
- Technical/practical role: defines steering profile model and enforcement metadata.

## 12. M6 - Evaluation and Deployment (Planned)

`evaluation/runner.py`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: reproducible evaluation orchestration boundary.
- Technical/practical role: runs benchmark datasets and reports quality/latency/cost metrics.

`evaluation/retrieval_metrics.py`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: retrieval-quality measurement boundary.
- Technical/practical role: computes retrieval metrics such as recall@k and ranking quality scores.

`evaluation/generation_metrics.py`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: generation-grounding measurement boundary.
- Technical/practical role: computes faithfulness and citation-grounding quality metrics.

`evaluation/datasets/*.jsonl`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: benchmark corpus definition boundary.
- Technical/practical role: stores fixed evaluation cases with versionable data.

`deployment/Dockerfile`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: environment portability artifact.
- Technical/practical role: defines container build and runtime setup.

`deployment/docker-compose.yml`
- Milestone metadata: introduced `M6`; status `planned`.
- Theoretical role: local deployment topology artifact.
- Technical/practical role: composes service and infrastructure for reproducible runs.

## 13. Update Protocol
- Add each new file under the milestone where it is first introduced.
- When behavior changes materially, update the file's milestone metadata and description.
- Keep `docs/03-didactic-traceability.md` focused on decisions and learning rationale; keep file mapping in this document.
