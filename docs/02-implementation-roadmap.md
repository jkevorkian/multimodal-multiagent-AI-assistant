# 02 - Implementation Roadmap

## 1. Roadmap Goal
Implement the assistant in milestone order so each phase is deployable, testable, and didactically traceable.

## 2. Milestone Order Overview
| Milestone | Name | Difficulty | Depends on |
| --- | --- | --- | --- |
| M0 | Skeleton and contracts | Medium | None |
| M1 | Text RAG baseline | Medium | M0 |
| M2 | ReAct tools + multi-agent orchestration | High | M1 |
| M3 | Image multimodal path | Medium | M2 |
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
Deliver ingest -> embed -> retrieve -> answer flow for text queries.

### Outputs
- Code modules:
  - `app/rag/ingestion.py`
  - `app/rag/chunking.py`
  - `app/rag/embeddings.py`
  - `app/rag/retriever.py`
  - `app/storage/pgvector_store.py`
- Endpoints:
  - `/ingest/documents`
  - `/query`
- Tests:
  - PDF and URL ingestion tests.
  - Chunking determinism tests.
  - Retrieval smoke tests with known corpus.
- Docs updates:
  - Add retrieval design and PGVector rationale in didactic doc.

### Risks and Mitigation
- Risk: Poor retrieval relevance due to chunking strategy.
- Mitigation: Add tunable chunk size/overlap and evaluate with fixed benchmark set.

### Rollback/Fallback
- Fallback to lexical retrieval (BM25/simple keyword) if embeddings unavailable.

### Definition of Done
- Query answers include retrieved citations.
- PGVector + JSONB metadata persistence works end-to-end.

## M2 - ReAct Tools + Multi-Agent Orchestration
### Objective
Add explicit role-based orchestration and tool routing.

### Outputs
- Code modules:
  - `app/agents/state.py`
  - `app/agents/research_agent.py`
  - `app/agents/analyst_agent.py`
  - `app/agents/answer_agent.py`
  - `app/agents/orchestrator.py`
  - `app/tools/registry.py`
- Endpoints:
  - `/agents/run`
- Tests:
  - State transition tests.
  - Tool-selection scenario tests.
  - Timeout/retry behavior tests.
- Docs updates:
  - Add agent control-flow trace example and state model explanation.

### Risks and Mitigation
- Risk: Unstable or looping agent behavior.
- Mitigation: Add max-step guardrails, tool budget, and fallback answer policy.

### Rollback/Fallback
- Route `/agents/run` to single-agent deterministic chain on orchestration failure.

### Definition of Done
- Multi-agent sequence executes with trace logs.
- Tool calls are bounded and observable.

## M3 - Image Multimodal Path
### Objective
Support image analysis and integrate visual findings into answer generation.

### Outputs
- Code modules:
  - `app/vision/preprocess.py`
  - `app/vision/adapter.py`
  - `app/vision/fusion.py`
- Endpoints:
  - `/vision/analyze`
- Tests:
  - Deterministic image fixtures.
  - Request size/type validation tests.
- Docs updates:
  - Add vision pipeline didactic section.

### Risks and Mitigation
- Risk: Hallucinated visual details.
- Mitigation: Force evidence-grounded response format with confidence notes.

### Rollback/Fallback
- If vision model unavailable, return explicit degraded-mode message and continue text flow.

### Definition of Done
- Image endpoint returns stable structured response with confidence and evidence summary.

## M4 - Video MVP Path
### Objective
Add video understanding with frame sampling + temporal aggregation.

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
  - `evaluation/datasets/*.jsonl`
  - `deployment/Dockerfile`
  - `deployment/docker-compose.yml`
- Endpoints:
  - `/metrics` includes aggregate benchmark results.
- Tests:
  - Reproducible benchmark run.
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
- Service runs from container and exposes public-ready config.

## 4. Cross-Milestone Quality Gates
- API contracts stay backward compatible unless explicitly versioned.
- Each milestone must add or update automated tests.
- Didactic traceability doc must be updated in the same PR as implementation.

## 5. Final Integration and Demo Sequence
1. Start service and validate `/health`.
2. Ingest a PDF and one URL with `/ingest/documents`.
3. Run text question through `/query` and inspect citations.
4. Run multi-agent execution through `/agents/run` and inspect trace.
5. Run `/vision/analyze` on sample image.
6. Run `/video/analyze` on sample short video.
7. Fetch `/metrics` and present latency/cost/accuracy summary.

## 6. Suggested Execution Rhythm
- One milestone per branch.
- Merge only when Definition of Done and quality gates are satisfied.
- After merge, immediately append technical decisions and lessons to didactic traceability.
