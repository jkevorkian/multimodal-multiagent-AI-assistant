from __future__ import annotations

from dataclasses import dataclass

from app.agents.state import AgentState
from app.core.context_compaction import ContextCompactionResult, ContextCompactor


@dataclass(frozen=True)
class ContextManagerResult:
    compacted: bool
    compaction: ContextCompactionResult


class AgentContextManager:
    def __init__(self, *, compactor: ContextCompactor, enabled: bool = True) -> None:
        self._compactor = compactor
        self._enabled = enabled

    def apply_pre_step_guard(self, state: AgentState) -> ContextManagerResult:
        if not self._enabled:
            passthrough = ContextCompactionResult(
                triggered=False,
                reason="disabled",
                previous_token_estimate=0,
                compacted_token_estimate=0,
            )
            return ContextManagerResult(compacted=False, compaction=passthrough)
        result = self._compactor.compact_if_needed(state)
        return ContextManagerResult(compacted=result.triggered, compaction=result)
