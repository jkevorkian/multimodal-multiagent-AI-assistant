from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    query: str
    trace: dict[str, str]
    allowed_tools: list[str]
    retrieval_filter: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 6
    tool_budget: int = 2
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    analysis_notes: list[str] = field(default_factory=list)
    context_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    final_answer: str = ""
    steps: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _step_count: int = 0

    def can_continue(self) -> bool:
        return self._step_count < self.max_steps

    @property
    def step_count(self) -> int:
        return self._step_count

    def record_step(self, step_name: str) -> None:
        self._step_count += 1
        self.steps.append(step_name)

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls.append(tool_name)

    def consume_tool_budget(self) -> bool:
        if self.tool_budget <= 0:
            return False
        self.tool_budget -= 1
        return True

    def clamp_confidence(self) -> None:
        self.confidence = max(0.0, min(0.99, float(self.confidence)))

    def snapshot(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "trace": dict(self.trace),
            "allowed_tools": list(self.allowed_tools),
            "retrieval_filter": dict(self.retrieval_filter),
            "max_steps": self.max_steps,
            "tool_budget": self.tool_budget,
            "retrieved_context": [dict(item) for item in self.retrieved_context],
            "tool_outputs": [dict(item) for item in self.tool_outputs],
            "analysis_notes": list(self.analysis_notes),
            "context_checkpoints": [dict(item) for item in self.context_checkpoints],
            "confidence": self.confidence,
            "final_answer": self.final_answer,
            "steps": list(self.steps),
            "tool_calls": list(self.tool_calls),
            "errors": list(self.errors),
            "step_count": self._step_count,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> AgentState:
        state = cls(
            query=str(snapshot.get("query", "")),
            trace=dict(snapshot.get("trace", {})),
            allowed_tools=list(snapshot.get("allowed_tools", [])),
            retrieval_filter=dict(snapshot.get("retrieval_filter", {})),
            max_steps=int(snapshot.get("max_steps", 6)),
            tool_budget=int(snapshot.get("tool_budget", 2)),
            retrieved_context=list(snapshot.get("retrieved_context", [])),
            tool_outputs=list(snapshot.get("tool_outputs", [])),
            analysis_notes=list(snapshot.get("analysis_notes", [])),
            context_checkpoints=list(snapshot.get("context_checkpoints", [])),
            confidence=float(snapshot.get("confidence", 0.0)),
            final_answer=str(snapshot.get("final_answer", "")),
            steps=list(snapshot.get("steps", [])),
            tool_calls=list(snapshot.get("tool_calls", [])),
            errors=list(snapshot.get("errors", [])),
        )
        state._step_count = int(snapshot.get("step_count", len(state.steps)))
        state.clamp_confidence()
        return state
