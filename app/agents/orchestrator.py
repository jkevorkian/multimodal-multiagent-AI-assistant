from __future__ import annotations

import time
from typing import Any
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.analyst_agent import AnalystAgent
from app.agents.answer_agent import AnswerAgent
from app.agents.checkpoint_store import NullCheckpointStore
from app.agents.context_manager import AgentContextManager
from app.agents.loop_controller import LoopController
from app.agents.research_agent import ResearchAgent
from app.agents.state import AgentState
from app.core.event_bus import InMemoryEventBus


class _OrchestratorGraphState(TypedDict):
    agent_state: AgentState


class AgentOrchestrator:
    def __init__(
        self,
        research_agent: ResearchAgent,
        analyst_agent: AnalystAgent,
        answer_agent: AnswerAgent,
        checkpoint_store: NullCheckpointStore | None = None,
        max_steps: int = 6,
        max_revision_iterations: int = 0,
        event_bus: InMemoryEventBus | None = None,
        run_timeout_sec: float = 30.0,
        context_manager: AgentContextManager | None = None,
    ) -> None:
        self._research_agent = research_agent
        self._analyst_agent = analyst_agent
        self._answer_agent = answer_agent
        self._checkpoint_store = checkpoint_store or NullCheckpointStore()
        self._max_steps = max_steps
        self._max_revision_iterations = max(0, int(max_revision_iterations))
        self._event_bus = event_bus
        self._run_timeout_sec = run_timeout_sec
        self._context_manager = context_manager
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_OrchestratorGraphState)
        graph.add_node("research_agent", self._run_research_node)
        graph.add_node("analyst_agent", self._run_analyst_node)
        graph.add_node("answer_agent", self._run_answer_node)
        graph.add_edge(START, "research_agent")
        graph.add_edge("research_agent", "analyst_agent")
        graph.add_edge("analyst_agent", "answer_agent")
        graph.add_edge("answer_agent", END)
        return graph.compile()

    async def _run_research_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "research_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_research_agent")
            return {"agent_state": state}
        await self._emit_step_event(
            state=state,
            event_type="agent.step.started",
            agent_name="research_agent",
            status_text="Researching and retrieving context...",
            metadata={"steps_completed": len(state.steps), "tool_budget_remaining": state.tool_budget},
        )
        state = await self._research_agent.run(state)
        await self._emit_step_event(
            state=state,
            event_type="agent.step.completed",
            agent_name="research_agent",
            status_text="Research step completed.",
            metadata={
                "steps_completed": len(state.steps),
                "retrieved_chunks": len(state.retrieved_context),
                "tool_calls_total": len(state.tool_calls),
                "retrieval_preview": self._build_retrieval_preview(state.retrieved_context),
                "retrieval_filter": dict(state.retrieval_filter),
            },
        )
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _run_analyst_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "analyst_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_analyst_agent")
            return {"agent_state": state}
        await self._emit_step_event(
            state=state,
            event_type="agent.step.started",
            agent_name="analyst_agent",
            status_text="Analyzing gathered evidence...",
            metadata={"steps_completed": len(state.steps), "retrieved_chunks": len(state.retrieved_context)},
        )
        state = await self._analyst_agent.run(state)
        await self._emit_step_event(
            state=state,
            event_type="agent.step.completed",
            agent_name="analyst_agent",
            status_text="Analysis step completed.",
            metadata={
                "steps_completed": len(state.steps),
                "analysis_notes": len(state.analysis_notes),
                "confidence": round(state.confidence, 4),
                "analysis_trace": state.analysis_notes[-5:],
            },
        )
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _run_answer_node(self, graph_state: _OrchestratorGraphState) -> _OrchestratorGraphState:
        state = graph_state["agent_state"]
        if "answer_agent" in state.steps:
            return {"agent_state": state}
        if not state.can_continue():
            state.errors.append("max_steps_reached_before_answer_agent")
            return {"agent_state": state}
        await self._emit_step_event(
            state=state,
            event_type="agent.step.started",
            agent_name="answer_agent",
            status_text="Composing final answer...",
            metadata={"steps_completed": len(state.steps), "context_chunks": len(state.retrieved_context)},
        )
        state = await self._answer_agent.run(state)
        await self._emit_step_event(
            state=state,
            event_type="agent.step.completed",
            agent_name="answer_agent",
            status_text="Answer synthesis step completed.",
            metadata={
                "steps_completed": len(state.steps),
                "confidence": round(state.confidence, 4),
                "answer_preview": state.final_answer[:320],
                "errors": state.errors[-3:],
            },
        )
        await self._save_checkpoint(state)
        return {"agent_state": state}

    async def _save_checkpoint(self, state: AgentState) -> None:
        trace_id = state.trace.get("trace_id", "unknown")
        await self._checkpoint_store.save(trace_id=trace_id, checkpoint=state.snapshot())

    async def run(
        self,
        query: str,
        trace: dict[str, str],
        allowed_tools: list[str],
        run_id: str | None = None,
        tool_budget: int = 2,
        max_steps: int | None = None,
        retrieval_top_k: int | None = None,
        resume_from_checkpoint: bool = False,
        retrieval_filter: dict[str, Any] | None = None,
    ) -> AgentState:
        started_at = time.perf_counter()
        trace_id = trace.get("trace_id", "unknown")
        resolved_run_id = run_id or trace.get("run_id") or trace_id
        resolved_max_steps = max_steps if max_steps is not None else self._max_steps
        max_revision_iterations = self._max_revision_iterations
        runtime_trace = dict(trace)
        runtime_trace["run_id"] = resolved_run_id
        runtime_trace["trace_id"] = trace_id
        if retrieval_top_k is not None:
            runtime_trace["retrieval_top_k"] = str(max(1, int(retrieval_top_k)))
        loop_controller = LoopController(
            max_steps=resolved_max_steps,
            max_tool_calls=tool_budget,
            timeout_budget_sec=self._run_timeout_sec,
        )
        await self._emit_event(
            run_id=resolved_run_id,
            trace_id=trace_id,
            event_type="run.started",
            status_text="Run started.",
            metadata={
                "query_summary": query[:120],
                "allowed_tools": allowed_tools,
                "max_steps": resolved_max_steps,
                "tool_budget": tool_budget,
                "max_revision_iterations": max_revision_iterations,
                "resume_from_checkpoint": resume_from_checkpoint,
                "retrieval_filter": dict(retrieval_filter or {}),
            },
        )
        state: AgentState
        if resume_from_checkpoint:
            latest = await self._checkpoint_store.latest(trace_id=trace_id)
            if latest and str(latest.get("query", "")) == query:
                state = AgentState.from_snapshot(latest)
                state.trace = runtime_trace
                await self._emit_event(
                    run_id=resolved_run_id,
                    trace_id=trace_id,
                    event_type="agent.revision.requested",
                    status_text="Resuming run from checkpoint snapshot.",
                    metadata={"reason_code": "resume_from_checkpoint", "resumed_steps": len(state.steps)},
                )
            else:
                state = AgentState(
                    query=query,
                    trace=runtime_trace,
                    allowed_tools=allowed_tools,
                    retrieval_filter=dict(retrieval_filter or {}),
                    tool_budget=tool_budget,
                    max_steps=resolved_max_steps,
                )
        else:
            state = AgentState(
                query=query,
                trace=runtime_trace,
                allowed_tools=allowed_tools,
                retrieval_filter=dict(retrieval_filter or {}),
                tool_budget=tool_budget,
                max_steps=resolved_max_steps,
            )

        state.max_steps = resolved_max_steps
        state.trace = runtime_trace
        state.allowed_tools = allowed_tools
        state.retrieval_filter = dict(retrieval_filter or state.retrieval_filter)
        revision_iteration = 0
        guardrail = None
        while True:
            if self._context_manager is not None:
                context_result = self._context_manager.apply_pre_step_guard(state)
                if context_result.compacted:
                    await self._emit_event(
                        run_id=resolved_run_id,
                        trace_id=trace_id,
                        event_type="agent.revision.requested",
                        status_text="Compacting context under token budget pressure.",
                        metadata={
                            "reason_code": "context_compaction",
                            "previous_token_estimate": context_result.compaction.previous_token_estimate,
                            "compacted_token_estimate": context_result.compaction.compacted_token_estimate,
                            "checkpoint_id": (
                                context_result.compaction.checkpoint.checkpoint_id
                                if context_result.compaction.checkpoint
                                else ""
                            ),
                        },
                    )
            state.trace["revision_iteration"] = str(revision_iteration)
            try:
                result = await self._graph.ainvoke({"agent_state": state})
                state = result["agent_state"]
            except Exception as exc:
                await self._emit_event(
                    run_id=resolved_run_id,
                    trace_id=trace_id,
                    event_type="run.failed",
                    status_text="Run failed due to internal error.",
                    metadata={"error": str(exc)},
                )
                raise

            loop_controller.record_state(state)
            elapsed_sec = time.perf_counter() - started_at
            guardrail = loop_controller.evaluate(state=state, elapsed_sec=elapsed_sec)
            if guardrail.triggered:
                break

            revision_reason = self._revision_reason(
                state=state,
                revision_iteration=revision_iteration,
                max_revision_iterations=max_revision_iterations,
            )
            if revision_reason is None:
                break

            await self._emit_event(
                run_id=resolved_run_id,
                trace_id=trace_id,
                event_type="agent.revision.requested",
                status_text="Requesting bounded revision iteration.",
                metadata={
                    "reason_code": revision_reason,
                    "iteration": revision_iteration + 1,
                    "remaining_steps": max(0, state.max_steps - state.step_count),
                },
            )
            revision_iteration += 1
            state = self._prepare_state_for_revision(
                state=state,
                revision_iteration=revision_iteration,
                revision_reason=revision_reason,
            )

        if not state.final_answer:
            state.final_answer = "Unable to produce final answer within current orchestration limits."
            state.confidence = min(state.confidence, 0.15)
            state.clamp_confidence()

        elapsed_sec = time.perf_counter() - started_at
        if guardrail is None:
            guardrail = loop_controller.evaluate(state=state, elapsed_sec=elapsed_sec)
        if guardrail.triggered:
            await self._emit_event(
                run_id=resolved_run_id,
                trace_id=trace_id,
                event_type="run.guardrail_triggered",
                status_text=f"Run stopped by guardrail: {guardrail.guard_name}",
                metadata={
                    "guard_name": guardrail.guard_name,
                    "reason": guardrail.reason or "",
                    "elapsed_sec": round(elapsed_sec, 4),
                    "steps": state.step_count,
                    "tool_calls": len(state.tool_calls),
                    "revision_iterations": revision_iteration,
                },
            )
        else:
            await self._emit_event(
                run_id=resolved_run_id,
                trace_id=trace_id,
                event_type="run.completed",
                status_text="Run completed successfully.",
                metadata={
                    "elapsed_sec": round(elapsed_sec, 4),
                    "steps": state.step_count,
                    "tool_calls": len(state.tool_calls),
                    "confidence": round(state.confidence, 4),
                    "revision_iterations": revision_iteration,
                },
            )
        return state

    def _revision_reason(
        self,
        *,
        state: AgentState,
        revision_iteration: int,
        max_revision_iterations: int,
    ) -> str | None:
        if revision_iteration >= max_revision_iterations:
            return None
        if not state.can_continue():
            return None

        normalized_answer = state.final_answer.lower()
        # Keep revisions for transient recoverable failures only.
        if state.errors and (
            "internal error" in normalized_answer
            or "failed" in normalized_answer
            or not state.final_answer.strip()
        ):
            return "error_recovery"
        return None

    def _prepare_state_for_revision(
        self,
        *,
        state: AgentState,
        revision_iteration: int,
        revision_reason: str,
    ) -> AgentState:
        # Clear per-pass markers so LangGraph nodes can execute again on bounded revisions.
        previous_steps = list(state.steps)
        state.steps = []
        state.final_answer = ""
        state.analysis_notes.append(
            (
                f"revision_iteration={revision_iteration};"
                f"reason={revision_reason};"
                f"prior_steps={previous_steps};"
                f"step_count={state.step_count}"
            )
        )
        return state

    async def _emit_step_event(
        self,
        *,
        state: AgentState,
        event_type: str,
        agent_name: str,
        status_text: str,
        metadata: dict,
    ) -> None:
        run_id = state.trace.get("run_id")
        trace_id = state.trace.get("trace_id", "unknown")
        if not run_id:
            return
        await self._emit_event(
            run_id=run_id,
            trace_id=trace_id,
            event_type=event_type,
            status_text=status_text,
            agent=agent_name,
            metadata=metadata,
        )

    async def _emit_event(
        self,
        *,
        run_id: str,
        trace_id: str,
        event_type: str,
        status_text: str,
        agent: str | None = None,
        tool: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            run_id=run_id,
            trace_id=trace_id,
            event_type=event_type,
            status_text=status_text,
            agent=agent,
            tool=tool,
            metadata=metadata or {},
        )

    @staticmethod
    def _build_retrieval_preview(retrieved_context: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
        preview: list[dict[str, Any]] = []
        for item in retrieved_context[:limit]:
            preview.append(
                {
                    "source": str(item.get("source", "unknown")),
                    "chunk_id": int(item.get("chunk_id", -1)),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "snippet": str(item.get("snippet", ""))[:220],
                }
            )
        return preview
