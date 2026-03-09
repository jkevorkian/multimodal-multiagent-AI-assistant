# 13 - Lead GenAI Interview Prep: Architecture Concepts from This Project

## 1. Purpose
This document summarizes the core concepts used in this repository and why they matter when designing a multimodal GenAI system with RAG and/or multi-agent orchestration.

## 2. Concept Map (High Level)
| Concept | How it appears in this project | Why it matters in architecture design |
| --- | --- | --- |
| Contract-first APIs | Typed request/response schemas in `app/contracts/*` | Keeps frontend, backend, and evaluation tooling aligned as complexity grows. |
| Interface boundaries | Provider protocols in `app/interfaces/*` | Lets you swap model/vector providers without rewriting business logic. |
| Dependency injection composition | `app/core/dependencies.py` service container | Centralizes runtime wiring and prevents route-level coupling. |
| Hybrid retrieval | Dense + lexical fusion in retriever | Improves recall and robustness across query styles (keyword and semantic). |
| Named vectors / multi-branch dense retrieval | `text_dense` + `mm_dense` in ingest/store/retriever | Supports cross-modal search while preserving text precision and debuggability. |
| Multimodal embedding abstraction | `app/rag/multimodal_embeddings.py` | Prevents lock-in and enables staged upgrades from fallback to true VL models. |
| Reranking stage | `app/rag/reranker.py` | Raises top-k precision and improves final grounding quality. |
| Evidence-grounded answering | Citations + retrieved chunks in query/agent responses | Reduces hallucination risk and gives inspectable provenance. |
| Modality-aware ingestion | `app/rag/ingestion.py` source type detection + media handling | Normalizes heterogeneous inputs into retrievable evidence units. |
| Video temporal evidence | Frame/timeline logic in `app/video/*` | Enables event-level reasoning instead of single-summary guessing. |
| Audio transcript indexing for video | Optional Whisper transcript during ingestion | Makes spoken content retrievable; critical for audio-heavy videos. |
| Metadata-filtered retrieval | `metadata_filter` + chat-scoped filters | Enforces context isolation (per chat/workspace/tenant) and reduces leakage. |
| Multi-agent role separation | Research/analyst/answer agents + orchestrator | Improves maintainability and observability of reasoning pipelines. |
| Tool registry and policy gating | `app/tools/registry.py` + steering policies | Constrains tool calls for safety, cost control, and deterministic behavior. |
| Checkpointing and resume | checkpoint store + run IDs | Supports long-running or interrupted workflows without recomputation. |
| Loop guardrails | step/tool/time budgets + stagnation controls | Prevents runaway autonomy loops and budget burn. |
| Runtime event telemetry | event bus + SSE `/runs/{run_id}/events` | Gives live observability for users/operators and supports replay/debugging. |
| Persistent chat memory | chat store + session/message/file APIs | Enables revisitable conversations with durable context boundaries. |
| Context compaction | `app/core/context_compaction.py` | Preserves key state while controlling context window growth and cost. |
| Steering controls | profile/tool/grounding controls in `app/core/steering.py` | Separates behavior policy from prompts; makes intent explicit and auditable. |
| Graceful degradation/fallback | deterministic/heuristic fallbacks across providers | Keeps the system operational when external dependencies fail. |
| Latency/cost budgets | top-k, frame caps, sample fps, token limits | Converts architecture into an SLO-aware system, not just a demo. |
| Storage portability | pgvector + Qdrant + fallback adapters | Supports local dev to production migration with minimal code churn. |
| Test-first milestone slices | milestone-focused tests under `tests/` | Prevents regressions while adding modalities, tools, and policies. |

## 3. Why These Concepts Matter for a Lead Role
At lead level, the key interview signal is not model trivia; it is whether you can design a system that is:
- Reliable under provider failures.
- Observable during runtime.
- Grounded and auditable for outputs.
- Evolvable as model/vendor capabilities change.
- Cost/latency-aware under real usage.
- Safe and controllable under autonomous behavior.

The concepts above are the architecture levers that make those properties possible.

## 4. Practical Design Heuristics You Can Explain in Interviews
1. Separate control plane from inference plane.
Control plane handles orchestration, policy, memory, and observability; inference plane handles model calls.

2. Keep retrieval as a pipeline, not a single call.
Candidate generation (multi-branch) then rerank, then answer with explicit evidence.

3. Treat modality as first-class metadata.
Evidence provenance (`source`, `modality`, `timestamp`, `frame`) should be preserved end-to-end.

4. Make safety and grounding configurable, not implicit.
Steering policies should be request/session-level inputs with deterministic precedence.

5. Design for degraded modes early.
If a multimodal provider is unavailable, still return useful grounded behavior from fallback paths.

6. Enforce budgets at each stage.
Tool budgets, step budgets, frame budgets, rerank pool budgets, and token budgets should all be explicit.

7. Build runtime introspection from day one.
Expose status/event streams and retrieval payloads so failures can be diagnosed quickly.

8. Optimize for replaceability.
Abstract model/vector/reranker providers so architecture survives future model shifts.

## 5. Suggested Study Order (External Deep Dive)
1. RAG retrieval architecture (hybrid retrieval, RRF, reranking, grounding metrics).
2. Multimodal retrieval design (shared vector space, named vectors, evidence schemas).
3. Agent orchestration patterns (state machines/graphs, tool policy, loop guards).
4. Context engineering for long sessions (memory layering + compaction).
5. Production operations (telemetry, SLOs, failure handling, rollback strategy).
