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

## 4.1 Concise Questions Answered (M0)
### Q1: Why define interfaces before real implementations?
A: It isolates behavior contracts early, so providers can be swapped without rewriting endpoint logic.

### Q2: Why use request/response schemas for stub endpoints?
A: Typed schemas force stable API contracts from the start and prevent accidental payload drift.

### Q3: Why introduce structured logging in M0 instead of later?
A: Traceability is cheapest when built early; retrofitting logs after multiple milestones is harder and less consistent.

### Q4: Why keep endpoint handlers thin?
A: Thin handlers reduce coupling and make test boundaries clearer (transport layer vs business logic).

### Q5: Why a service container in M0?
A: It gives a single composition root for dependencies and makes dependency injection explicit and testable.

### Q6: Why use stubs now?
A: Stubs let contracts and orchestration surfaces be validated before expensive integrations exist.

### Q7: Why expose all required endpoints now as placeholders?
A: Early contract visibility enables client and test development in parallel with backend feature implementation.

### Q8: Why include request and trace IDs in every response path?
A: They create a minimal observability backbone for debugging cross-component flows in later milestones.

### Q9: Why is this still useful if answers are placeholders?
A: Milestone 0 validates architecture quality, not model quality. It de-risks integration and delivery order.

### Q10: What is the main tradeoff of this approach?
A: Slightly more upfront structure in exchange for lower refactor cost during RAG, agents, and multimodal expansion.

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
