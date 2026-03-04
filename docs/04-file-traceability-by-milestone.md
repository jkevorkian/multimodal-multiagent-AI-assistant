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
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: minimal dependency manifest for quick bootstrap.
- Technical/practical role: provides base package list for lightweight environment setup.

`pyproject.toml`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: canonical packaging and tooling configuration.
- Technical/practical role: declares runtime/dev dependencies and pytest defaults.

`docs/01-requirements.md`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: source of truth for system scope and acceptance criteria.
- Technical/practical role: defines FR/NFR testable requirements and boundaries.

`docs/02-implementation-roadmap.md`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: milestone planning artifact for incremental delivery.
- Technical/practical role: defines objectives, outputs, risks, and DoD per milestone.

`docs/03-didactic-traceability.md`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: learning/decision narrative over implementation lifecycle.
- Technical/practical role: stores didactic cards, decision log, and troubleshooting heuristics.

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
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: centralized environment-driven configuration contract.
- Technical/practical role: defines `Settings`, including app and RAG tuning parameters.

`app/core/logging.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: observability boundary for request lifecycle tracing.
- Technical/practical role: configures JSON logging and injects request/trace IDs.

`app/core/dependencies.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: dependency inversion entrypoint between API and services.
- Technical/practical role: builds and provides `ServiceContainer` for endpoint dependencies.

### 3.4 API Contracts
`app/contracts/__init__.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: schema package boundary marker.
- Technical/practical role: keeps request/response contract modules isolated.

`app/contracts/schemas.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: schema-first API contract.
- Technical/practical role: defines request/response models and validation constraints.

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
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: future multi-agent orchestration entrypoint contract.
- Technical/practical role: implements `POST /agents/run` placeholder flow.

`app/api/routes/vision.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: image multimodal entrypoint contract.
- Technical/practical role: implements `POST /vision/analyze` placeholder flow.

`app/api/routes/video.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: video multimodal entrypoint contract.
- Technical/practical role: implements `POST /video/analyze` placeholder flow.

`app/api/routes/metrics.py`
- Milestone metadata: introduced `M0`; updated `M0`; status `implemented`.
- Theoretical role: operational metrics contract boundary.
- Technical/practical role: implements `GET /metrics`.

### 3.7 Test Layer
`tests/test_api_contracts.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role: contract-level regression guard.
- Technical/practical role: validates route presence, response schema shape, and trace headers.

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
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: embedding engine for semantic similarity.
- Technical/practical role: implements pluggable embedding adapters (deterministic, OpenAI, SentenceTransformers), batch embedding support, and cosine similarity.

`app/rag/ingestion.py`
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
- Theoretical role: ingestion orchestration boundary for document indexing.
- Technical/practical role: loads URL/file sources, extracts text, chunks, embeds, and upserts metadata.

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
- Milestone metadata: introduced `M1`; updated `M1`; status `implemented`.
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
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: extends API contract to expose indexing result granularity.
- Technical/practical change: adds `indexed_chunks` to ingest response.

`app/api/routes/ingest.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: transitions from placeholder contract to executable ingestion boundary.
- Technical/practical change: calls ingestion service and returns indexed chunk totals.

`app/api/routes/query.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: transitions from stub retrieval flow to grounded citation-producing flow.
- Technical/practical change: executes retrieval over indexed chunks and returns chunk-level citations.

`tests/test_api_contracts.py`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: contract checks now cover M1 ingest output semantics.
- Technical/practical change: ingest test validates non-zero indexed chunk count for fixture data.

`docs/03-didactic-traceability.md`
- Milestone metadata: introduced `M0`; updated `M1`; status `implemented`.
- Theoretical role change: remains decision/learning artifact; file-level mapping moved out.
- Technical/practical change: references this document as the source for per-file traceability.

## 5. M2 - ReAct Tools + Multi-Agent Orchestration (Planned)

`app/agents/state.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: shared state model for multi-agent coordination.
- Technical/practical role: defines mandatory state keys and transition payload shape.

`app/agents/research_agent.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: evidence-gathering agent role.
- Technical/practical role: executes retrieval/tool actions and writes research artifacts to shared state.

`app/agents/analyst_agent.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: intermediate synthesis role.
- Technical/practical role: transforms research outputs into analysis notes and confidence cues.

`app/agents/answer_agent.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: final composition role.
- Technical/practical role: generates grounded response from shared state artifacts.

`app/agents/orchestrator.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: control-flow coordinator for role sequencing and guardrails.
- Technical/practical role: enforces bounded loops, transition checks, and termination policy.

`app/tools/registry.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: tool registration and resolution boundary.
- Technical/practical role: maps tool IDs to executable implementations and selection policies.

`app/tools/mcp_adapter.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: protocol bridge between internal tool system and MCP-compliant servers.
- Technical/practical role: converts MCP tool descriptors/invocations into internal tool registry contracts.

`app/agents/checkpoint_store.py`
- Milestone metadata: introduced `M2`; status `planned`.
- Theoretical role: durability boundary for long-running agent workflows.
- Technical/practical role: persists and restores orchestration checkpoints for resume-safe execution.

## 6. M3 - Image Multimodal Path (Planned)

`app/vision/preprocess.py`
- Milestone metadata: introduced `M3`; status `planned`.
- Theoretical role: preprocessing boundary for image quality normalization.
- Technical/practical role: validates image input and prepares payloads for inference.

`app/vision/adapter.py`
- Milestone metadata: introduced `M3`; status `planned`.
- Theoretical role: provider abstraction realization for vision inference.
- Technical/practical role: invokes vision-capable client and normalizes provider-specific output.

`app/vision/fusion.py`
- Milestone metadata: introduced `M3`; status `planned`.
- Theoretical role: multimodal fusion boundary between visual and textual evidence.
- Technical/practical role: merges visual findings into answer-generation context.

## 7. M4 - Video MVP Path (Planned)

`app/video/frame_sampler.py`
- Milestone metadata: introduced `M4`; status `planned`.
- Theoretical role: temporal sampling strategy boundary.
- Technical/practical role: selects representative frames under latency/cost budget.

`app/video/temporal_aggregator.py`
- Milestone metadata: introduced `M4`; status `planned`.
- Theoretical role: temporal reasoning and synthesis boundary.
- Technical/practical role: combines frame-level findings into sequence-aware summary.

`app/video/adapter.py`
- Milestone metadata: introduced `M4`; status `planned`.
- Theoretical role: provider abstraction realization for video inference.
- Technical/practical role: coordinates frame/video analysis and normalizes event outputs.

## 8. M5 - Production Hardening (Planned)

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

## 9. M6 - Evaluation and Deployment (Planned)

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

## 10. Update Protocol
- Add each new file under the milestone where it is first introduced.
- When behavior changes materially, update the file's milestone metadata and description.
- Keep `docs/03-didactic-traceability.md` focused on decisions and learning rationale; keep file mapping in this document.
