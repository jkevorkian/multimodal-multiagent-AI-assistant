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
| M3 | Image multimodal path | Medium | M2.2 |
| M4 | Video MVP path | High | M3 |
| M5 | Production hardening | High | M1-M4 |
| M6 | Evaluation and deployment | Medium | M5 |

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
