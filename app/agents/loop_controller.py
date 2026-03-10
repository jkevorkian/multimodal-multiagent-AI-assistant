from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from app.agents.state import AgentState


@dataclass(frozen=True)
class GuardrailDecision:
    triggered: bool
    guard_name: str | None = None
    reason: str | None = None


class LoopController:
    def __init__(
        self,
        max_steps: int,
        max_tool_calls: int,
        stagnation_window: int = 4,
        timeout_budget_sec: float = 120.0,
    ) -> None:
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._timeout_budget_sec = timeout_budget_sec
        self._recent_state_signatures: deque[str] = deque(maxlen=max(stagnation_window, 2))

    def record_state(self, state: AgentState) -> None:
        signature = "|".join(
            (
                str(state.step_count),
                str(len(state.tool_calls)),
                str(len(state.analysis_notes)),
                str(len(state.errors)),
                str(bool(state.final_answer.strip())),
            )
        )
        self._recent_state_signatures.append(signature)

    def evaluate(self, state: AgentState, elapsed_sec: float) -> GuardrailDecision:
        if elapsed_sec >= self._timeout_budget_sec:
            return GuardrailDecision(
                triggered=True,
                guard_name="timeout_budget",
                reason=f"elapsed_sec={elapsed_sec:.3f} exceeded timeout_budget_sec={self._timeout_budget_sec:.3f}",
            )

        if state.step_count >= self._max_steps and not state.final_answer.strip():
            return GuardrailDecision(
                triggered=True,
                guard_name="max_steps",
                reason=f"steps={state.step_count} reached max_steps={self._max_steps}",
            )

        if len(state.tool_calls) >= self._max_tool_calls and state.tool_budget <= 0 and not state.final_answer.strip():
            return GuardrailDecision(
                triggered=True,
                guard_name="max_tool_calls",
                reason=f"tool_calls={len(state.tool_calls)} reached max_tool_calls={self._max_tool_calls}",
            )

        if (
            len(self._recent_state_signatures) == self._recent_state_signatures.maxlen
            and len(set(self._recent_state_signatures)) == 1
        ):
            return GuardrailDecision(
                triggered=True,
                guard_name="stagnation_detector",
                reason="state signature repeated without meaningful delta",
            )

        return GuardrailDecision(triggered=False)
