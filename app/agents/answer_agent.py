from __future__ import annotations

from app.agents.state import AgentState
from app.interfaces.llm import LLMClient


class AnswerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: AgentState) -> AgentState:
        state.record_step("answer_agent")
        context_parts = [item.get("snippet", "") for item in state.retrieved_context[:5] if item.get("snippet")]
        context_parts.extend(note for note in state.analysis_notes[:4])

        try:
            state.final_answer = await self._llm.generate(state.query, context_parts)
        except Exception as exc:
            state.errors.append(f"answer_error:{exc}")
            state.final_answer = "I could not complete answer synthesis due to an internal error."

        if not state.final_answer.strip():
            state.final_answer = "I do not have enough evidence yet to answer confidently."
            state.confidence = min(state.confidence, 0.2)
        state.clamp_confidence()
        return state
