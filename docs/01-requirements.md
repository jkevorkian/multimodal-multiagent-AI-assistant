# 01 - Requirements

## 1. Purpose
This document defines what the `multimodal-multiagent-AI-assistant` must do, what is intentionally out of scope, and how completion is measured.

## 2. Non-goals
- Domain-specific crypto analytics or crypto database integration.
- Web UI in MVP (CLI-first delivery is mandatory).
- Model fine-tuning in MVP.
- Enterprise IAM/SSO and multi-tenant authorization in MVP.

## 3. Scope Boundaries
### In scope
- Document consultation and Q&A with RAG.
- Tool-using agent behavior (ReAct style).
- Multi-agent orchestration with explicit role separation.
- Multimodal understanding for text, image, and video.
- FastAPI backend with async endpoints and production-oriented error handling.
- Evaluation, cost optimization, and deployable packaging.

### Out of scope
- Crypto-specific use cases, schemas, or data pipelines.
- Rich browser frontend before core backend milestones are stable.

## 4. Functional Requirements (Testable)
| ID | Requirement | Verification method |
| --- | --- | --- |
| FR-001 | The system shall ingest documents from PDF files and web URLs. | Integration test that ingests sample PDF and URL and confirms indexed chunks > 0. |
| FR-002 | The system shall chunk ingested content and store chunk metadata for retrieval traceability. | Unit test validating chunk boundaries and metadata fields (`source`, `chunk_id`, `offset`). |
| FR-003 | The system shall support pluggable embedding providers and use a neural-network embedding model in production mode. | Adapter tests for provider selection + integration test that indexes and retrieves semantically related text. |
| FR-004 | The system shall persist vectors in PostgreSQL with PGVector and JSONB metadata. | DB integration test confirming vector and metadata insertion/retrieval. |
| FR-005 | The system shall use hybrid retrieval (dense + lexical) with rank fusion and optional reranking. | Retrieval test asserting relevant chunk is retrieved when either lexical or semantic branch is strong. |
| FR-006 | The system shall answer text queries using retrieved context (RAG) with citations. | End-to-end test with known question/context pair and citation presence. |
| FR-007 | The system shall expose tool-calling behavior where the agent decides when to call tools. | Scenario test where at least one tool is selected and executed by policy. |
| FR-008 | The system shall implement multi-agent roles: Research Agent, Analyst Agent, Answer Agent. | Orchestration test verifying role execution order and shared state transitions. |
| FR-009 | The system shall analyze images and return a textual interpretation. | API test for `/vision/analyze` using deterministic sample image. |
| FR-010 | The system shall analyze video input using frame sampling and temporal aggregation. | API test for `/video/analyze` validating frame extraction and synthesis output. |
| FR-011 | The system shall expose the following API endpoints: `/health`, `/ingest/documents`, `/query`, `/agents/run`, `/agents/tools`, `/vision/analyze`, `/video/analyze`, `/metrics`. | Contract tests confirming endpoint availability, status codes, and schema conformance. |
| FR-012 | The system shall provide evaluation outputs for accuracy, latency, and cost at run and aggregate levels. | Evaluation runner test generating per-run and summary report artifacts. |
| FR-013 | The system shall support pluggable vector store backends with fallback coexistence (external vector DB + PostgreSQL/in-memory). | Adapter tests validating fallback behavior when primary store is unavailable. |

## 5. Required Public Contracts
### 5.1 Endpoints
- `GET /health`
- `POST /ingest/documents`
- `POST /query`
- `POST /agents/run`
- `GET /agents/tools`
- `POST /vision/analyze`
- `POST /video/analyze`
- `GET /metrics`

### 5.2 Core Schemas (Pydantic)
- `IngestRequest`, `IngestResponse`
- `QueryRequest`, `QueryResponse`
- `AgentRunRequest`, `AgentRunResponse`
- `AgentToolsResponse`
- `VisionRequest`, `VisionResponse`
- `VideoRequest`, `VideoResponse`
- `MetricsResponse`

### 5.3 Core Interfaces
- `LLMClient`
- `EmbeddingClient`
- `VisionClient`
- `VideoClient`
- `VectorStore`
- `Retriever`
- `Reranker`
- `Tool`

### 5.4 Orchestration State
Shared state object must include at least:
- `query`
- `retrieved_context`
- `tool_outputs`
- `analysis_notes`
- `confidence`
- `final_answer`
- `trace`

## 6. Non-Functional Requirements
### 6.1 Performance
- NFR-PERF-001: `/query` text response p95 latency <= 8s for corpus up to 10,000 chunks in MVP test environment.
- NFR-PERF-002: `/vision/analyze` p95 latency <= 10s for image <= 5MB.
- NFR-PERF-003: `/video/analyze` p95 latency <= 30s for video <= 30s duration and <= 100MB.
- NFR-PERF-004: retrieval stage shall support independent budgets for dense search, lexical search, and reranking candidate pool.

### 6.2 Reliability
- NFR-REL-001: >= 99% successful responses over 100-request smoke run excluding invalid input cases.
- NFR-REL-002: External model/API calls must use bounded retries with exponential backoff.

### 6.3 Observability
- NFR-OBS-001: Structured logs must include `timestamp`, `level`, `request_id`, `trace_id`, `component`, `latency_ms`.
- NFR-OBS-002: Pipeline stage events (ingest, retrieve, tool_call, compose_answer) must be traceable in logs.

### 6.4 Cost Constraints
- NFR-COST-001: Embedding and response caching must be enabled by default in non-debug environments.
- NFR-COST-002: Model routing must support at least one lower-cost default model and one higher-quality fallback model.
- NFR-COST-003: embedding provider selection must allow low-cost vs high-quality profiles without code changes.

### 6.5 Security Baseline
- NFR-SEC-001: Secrets must be sourced from environment variables; no hardcoded API keys.
- NFR-SEC-002: Uploaded files must be validated by MIME type and max size before processing.
- NFR-SEC-003: External URL ingestion must include allowlist or safe-fetch controls.

## 7. MVP Definition
MVP is complete when all of the following are true:
- Text RAG works end-to-end with citations.
- ReAct-style tool invocation works through multi-agent orchestration.
- Image analysis endpoint is functional.
- Video analysis endpoint is functional with documented frame-sampling strategy.
- Evaluation reports include accuracy, latency, and cost.
- System is containerized and can be deployed to a public endpoint.

## 8. Acceptance Criteria Checklist
### Functional
- [ ] FR-001 through FR-013 are validated by automated tests.
- [ ] API contract tests pass for all required endpoints.
- [ ] Multi-agent run traces show role-specific steps and state transitions.

### Non-functional
- [ ] NFR-PERF thresholds are met in benchmark report.
- [ ] NFR-REL thresholds are met in reliability smoke test.
- [ ] NFR-OBS fields are present in structured logs.
- [ ] NFR-COST controls are active and measurable.
- [ ] NFR-SEC baseline checks pass for ingestion and config handling.

## 9. Change Control
Any requirement change must update:
- This document (`01-requirements.md`).
- Milestone scope in `02-implementation-roadmap.md`.
- Related rationale entry in `03-didactic-traceability.md`.
