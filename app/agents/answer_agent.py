from __future__ import annotations

from app.agents.state import AgentState
from app.core.event_bus import InMemoryEventBus
from app.interfaces.llm import LLMClient


class AnswerAgent:
    def __init__(self, llm: LLMClient, event_bus: InMemoryEventBus | None = None) -> None:
        self._llm = llm
        self._event_bus = event_bus

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("answer_agent")
        context_parts = [item.get("snippet", "") for item in state.retrieved_context[:5] if item.get("snippet")]
        context_parts.extend(note for note in state.analysis_notes[:4])

        try:
            await self._emit_model_call_event(state)
            state.final_answer = await self._llm.generate(state.query, context_parts)
        except Exception as exc:
            state.errors.append(f"answer_error:{exc}")
            state.final_answer = "I could not complete answer synthesis due to an internal error."

        if not state.final_answer.strip():
            state.final_answer = "I do not have enough evidence yet to answer confidently."
            state.confidence = min(state.confidence, 0.2)
        state.clamp_confidence()
        return state

    async def _emit_model_call_event(self, state: AgentState) -> None:
        if self._event_bus is None:
            return
        run_id = state.trace.get("run_id")
        trace_id = state.trace.get("trace_id", "unknown")
        if not run_id:
            return
        await self._event_bus.emit(
            run_id=run_id,
            trace_id=trace_id,
            event_type="model.call.in_progress",
            status_text="Synthesizing final answer...",
            agent="answer_agent",
            metadata={"model": self._llm.__class__.__name__},
        )
