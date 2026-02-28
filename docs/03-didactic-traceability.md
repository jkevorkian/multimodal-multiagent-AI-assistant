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
| M1 | Text RAG | Chunking, embeddings, similarity search, retrieval grounding | PostgreSQL, PGVector, JSONB |
| M2 | Multi-agent ReAct | Agent roles, shared state, tool routing, bounded reasoning loops | LangGraph, tool registry patterns |
| M3 | Image path | Input preprocessing, multimodal inference, evidence fusion | Vision model adapters |
| M4 | Video path | Frame sampling, temporal aggregation, budget-aware processing | Video pipeline utilities |
| M5 | Hardening | Retry strategy, caching layers, model routing, error taxonomy | Async IO, resilient API patterns |
| M6 | Evaluation + deploy | Metric design, reproducibility, containerization | Benchmark runners, Docker |

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
Documents are ingested, chunked, embedded, and stored as vectors. Query vectors retrieve nearest chunks, and the model answers using those chunks.

### Why it works
Semantic retrieval approximates conceptual similarity better than plain keyword matching, improving grounded answers.

### Why this technology fits
PGVector inside PostgreSQL keeps vectors and metadata in one operational store, simplifying early architecture.

## M2 - ReAct + Multi-agent
### How it works
A shared state object is passed through role-specific agents (research, analyst, answer). Agents call tools when needed and write intermediate artifacts to state.

### Why it works
Role decomposition narrows each reasoning step, improving controllability and debugging compared with one monolithic prompt.

### Why this technology fits
LangGraph provides explicit control flow and state transitions, which are ideal for traceable multi-agent pipelines.

## M3 - Image Multimodal
### How it works
Image input is validated and preprocessed, then sent to a vision-capable adapter. Extracted signals are fused into textual reasoning.

### Why it works
Grounding answer generation in visual evidence reduces unsupported statements.

### Why this technology fits
A vision adapter abstraction allows swapping providers while preserving endpoint contracts.

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

## 7. Common Failure Modes and Debugging Heuristics

## RAG
- Failure mode: low-relevance retrieval.
- Heuristic: inspect chunk size/overlap, embedding model choice, and top-k configuration.

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

## 9. M0 File-by-File Traceability (Theory + Practice)

## 9.1 Project Setup
`pyproject.toml`
- Theoretical role: defines the project as a reproducible Python package with explicit dependency boundaries.
- Technical/practical role: declares runtime deps (FastAPI, Pydantic), dev deps (pytest/httpx), and pytest defaults.

## 9.2 Application Composition Root
`app/__init__.py`
- Theoretical role: marks `app` as the root application package.
- Technical/practical role: enables absolute imports like `from app.main import app`.

`app/main.py`
- Theoretical role: composition root that wires configuration, middleware, and route graph.
- Technical/practical role: creates the FastAPI app via `create_app()`, configures JSON logging, adds request-context middleware, includes routers, and emits startup log event.

## 9.3 Core Infrastructure
`app/core/__init__.py`
- Theoretical role: package marker for core cross-cutting concerns.
- Technical/practical role: groups config/logging/dependency modules under a stable namespace.

`app/core/config.py`
- Theoretical role: centralizes runtime configuration to support environment-driven behavior.
- Technical/practical role: defines `Settings` via `BaseSettings` with `MMAA_` env prefix and exposes `settings` singleton.

`app/core/logging.py`
- Theoretical role: implements observability as a first-class architecture concern.
- Technical/practical role: provides `JsonFormatter`, root logger setup, and `RequestContextMiddleware` that injects `x-request-id`/`x-trace-id`, measures latency, and logs request completion.

`app/core/dependencies.py`
- Theoretical role: dependency inversion entry point, decoupling API layer from concrete providers.
- Technical/practical role: defines stub implementations (`StubLLMClient`, `StubVisionClient`, etc.), `ServiceContainer` dataclass, and cached `get_container()` provider for FastAPI `Depends`.

## 9.4 API Contracts
`app/contracts/__init__.py`
- Theoretical role: marks schema package as API contract domain.
- Technical/practical role: keeps import boundaries clean for request/response models.

`app/contracts/schemas.py`
- Theoretical role: schema-first API design that stabilizes interfaces before business logic.
- Technical/practical role: defines all request/response models (`IngestRequest`, `QueryResponse`, `VideoResponse`, `MetricsResponse`, etc.) with validation constraints (lengths, ranges, required fields).

## 9.5 Provider Abstraction Interfaces
`app/interfaces/__init__.py`
- Theoretical role: single export surface for interface contracts.
- Technical/practical role: re-exports protocol types to simplify imports in other layers.

`app/interfaces/llm.py`
- Theoretical role: standard contract for text generation behavior.
- Technical/practical role: declares `LLMClient.generate(prompt, context)` protocol.

`app/interfaces/embedding.py`
- Theoretical role: abstraction for vectorization capability.
- Technical/practical role: declares `EmbeddingClient.embed_text(text)` returning `list[float]`.

`app/interfaces/vision.py`
- Theoretical role: model-agnostic visual reasoning contract.
- Technical/practical role: declares `VisionClient.analyze_image(image_uri, prompt)`.

`app/interfaces/video.py`
- Theoretical role: separates temporal multimodal behavior from specific vendor APIs.
- Technical/practical role: declares `VideoClient.analyze_video(...)` with sampling parameters (`sample_fps`, `max_frames`).

`app/interfaces/vector_store.py`
- Theoretical role: persistence boundary for vector DB concerns.
- Technical/practical role: declares `upsert()` and `search()` operations used by retrieval systems.

`app/interfaces/retriever.py`
- Theoretical role: retrieval strategy abstraction independent of storage implementation.
- Technical/practical role: declares `retrieve(query, top_k)` contract.

`app/interfaces/tool.py`
- Theoretical role: pluggable tool contract for ReAct-style agent actions.
- Technical/practical role: declares tool `name` and async `run(payload)` signature.

## 9.6 API Layer
`app/api/__init__.py`
- Theoretical role: package marker for transport layer concerns.
- Technical/practical role: groups routers and route modules.

`app/api/router.py`
- Theoretical role: central route composition entrypoint.
- Technical/practical role: builds `api_router` and includes all required route modules so endpoint registration is deterministic.

`app/api/routes/__init__.py`
- Theoretical role: route namespace marker.
- Technical/practical role: enables clean module imports from `app.api.routes`.

`app/api/routes/health.py`
- Theoretical role: operational liveness/readiness contract.
- Technical/practical role: exposes `GET /health` returning service status, name, and version.

`app/api/routes/ingest.py`
- Theoretical role: ingestion boundary between external content sources and future indexing pipeline.
- Technical/practical role: exposes `POST /ingest/documents`, validates payload, returns accepted source count and trace metadata.

`app/api/routes/query.py`
- Theoretical role: baseline RAG query interaction boundary.
- Technical/practical role: exposes `POST /query`, calls retriever + LLM stubs, returns answer/citations/confidence/trace.

`app/api/routes/agents.py`
- Theoretical role: orchestration entrypoint for multi-agent behavior.
- Technical/practical role: exposes `POST /agents/run`, returns role-step sequence and tool-call list.

`app/api/routes/vision.py`
- Theoretical role: image modality integration boundary.
- Technical/practical role: exposes `POST /vision/analyze`, invokes vision client stub, returns findings and confidence.

`app/api/routes/video.py`
- Theoretical role: video modality integration boundary with explicit temporal-processing knobs.
- Technical/practical role: exposes `POST /video/analyze`, forwards sampling params, returns key events and processed frame count.

`app/api/routes/metrics.py`
- Theoretical role: observability and evaluation reporting boundary.
- Technical/practical role: exposes `GET /metrics` with placeholder latency/cost/accuracy fields and timestamp.

## 9.7 Test Layer
`tests/test_api_contracts.py`
- Theoretical role: contract verification layer ensuring the architecture shell is correct before real feature logic.
- Technical/practical role: validates required routes, schema-shaped responses, and request/trace headers using FastAPI `TestClient`.
