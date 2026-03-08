# 11 - Live Status + Loop Orchestration Plan (M2.3)

## 1. Goal
Define a production-style implementation plan for:
- live user-facing runtime status ("thinking", "processing", "tool running", "revising")
- revision-capable multi-agent loops
- deterministic anti-infinite-loop guardrails

This plan keeps API compatibility with existing `/query` and `/agents/run` contracts.
Follow-up extension for durable chat sessions and transcript-grade runtime detail is documented in `docs/12-chat-persistence-and-runtime-reasoning-plan.md` (`M2.4`).

## 2. Industrial Patterns Observed (Primary Sources)

### 2.1 Evented Runtime Visibility
- OpenAI Responses streaming defines granular server events with ordered `sequence_number` fields and explicit lifecycle events (`response.in_progress`, `response.completed`, `response.failed`, `response.incomplete`) plus deltas for text and tool arguments.
- OpenAI Agents SDK tracing models runs as traces/spans and captures agent runs, LLM generations, tool calls, handoffs, and guardrails by default.
- AWS Bedrock agent invocation supports `enableTrace=True` and emits trace events during streaming responses.
- Vertex Agent Engine positions tracing/logging/monitoring as first-class production observability surfaces.

### 2.2 Loop + Revision Safety
- Anthropic's evaluator-optimizer workflow uses explicit generation/evaluation loops where feedback drives iterative refinement.
- AutoGen operationalizes bounded loops with composable termination conditions (max messages, token usage, text mention, AND/OR composition).
- LangGraph explicitly documents recursion-limit errors and recommends step limits (`recursion_limit`) to bound cyclic graphs.

## 3. Proposed Runtime Event Contract

### 3.1 Event Envelope
- `run_id`: stable runtime ID
- `trace_id`: observability correlation ID
- `sequence_number`: monotonic event sequence per run
- `timestamp`: UTC ISO-8601
- `event_type`: enum (below)
- `status_text`: user-facing high-level text
- `agent`: optional (`research_agent`, `analyst_agent`, `answer_agent`, `critic_agent`)
- `tool`: optional tool name
- `metadata`: bounded structured fields (latency, retries, budget remaining, guard name)

### 3.2 Event Types (Minimal Set)
- `run.started`
- `agent.step.started`
- `agent.step.completed`
- `tool.call.started`
- `tool.call.completed`
- `tool.call.failed`
- `model.call.in_progress`
- `agent.revision.requested`
- `run.completed`
- `run.failed`
- `run.guardrail_triggered`

## 4. Loop/Revision Graph Design

### 4.1 Flow
- Planner/Research -> Executor -> Critic -> Revision Gate
- Gate decisions:
  - approve -> Answer -> End
  - revise -> Planner/Research (loop)
  - guardrail triggered -> Safe Exit (partial answer + diagnostics)

### 4.2 Deterministic Guardrails
- `max_steps`: hard cap on loop iterations
- `max_tool_calls`: cap tool amplification
- `stagnation_detector`: break if no meaningful state delta for N loops
- `timeout_budget`: wall-clock SLA bound

### 4.3 Safe Exit Behavior
- Return best-effort grounded output from current evidence.
- Attach `guardrail_triggered` reason and budget stats in trace metadata.
- Never silently terminate.

## 5. Backend Implementation Slice (M2.3 Minimal)

### 5.1 Modules
- `app/contracts/runtime_events.py`
- `app/core/event_bus.py`
- `app/agents/loop_controller.py`
- `app/agents/progress_mapper.py`

### 5.2 Endpoints
- `GET /runs/{run_id}/events` (SSE stream)
- `GET /runs/{run_id}/status` (latest snapshot)

### 5.3 Wiring Points
- Emit events at:
  - orchestrator stage start/end
  - LLM call start/end/error
  - tool call start/end/error
  - revision-loop transitions
  - guardrail terminations

## 6. Frontend Behavior
- Live banner cycles through status text mapped from events.
- Timeline panel appends ordered events with stage/tool badges.
- Agent cards reflect current status (`idle`, `running`, `done`, `failed`, `guardrail_exit`).
- Tool log shows tool name, status, latency, retries.

## 7. Testing and Acceptance

### 7.1 Tests
- Event schema and ordering tests.
- SSE stream tests (reconnect + last-event-id resume).
- Guardrail unit tests.
- Orchestrator integration tests with forced loop scenarios.

### 7.2 Definition of Done
- Users can see meaningful in-flight status without waiting for final answer.
- Revision loops improve output while staying budget-bounded.
- Infinite loops are prevented by policy, not by chance.

## 8. Primary References
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph recursion limit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
- OpenAI Responses streaming events: https://platform.openai.com/docs/api-reference/responses-streaming
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- Anthropic evaluator-optimizer workflow: https://www.anthropic.com/engineering/building-effective-agents
- AutoGen termination conditions: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html
- Amazon Bedrock agent tracing (`enableTrace`): https://docs.aws.amazon.com/bedrock/latest/userguide/agents-invoke-agent.html
- Vertex Agent Engine observability: https://docs.cloud.google.com/agent-builder/agent-engine/overview
