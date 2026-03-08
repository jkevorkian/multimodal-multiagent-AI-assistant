from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.agents.state import AgentState
from app.contracts.context import ContextCheckpoint, PinnedContext


def estimate_text_tokens(text: str) -> int:
    # Practical approximation for budget control without external tokenizer dependency.
    return max(1, len(text) // 4) if text else 0


def estimate_agent_state_tokens(state: AgentState) -> int:
    total = estimate_text_tokens(state.query)
    total += sum(estimate_text_tokens(note) for note in state.analysis_notes)
    total += sum(estimate_text_tokens(str(item.get("snippet", ""))) for item in state.retrieved_context)
    total += sum(estimate_text_tokens(str(item)) for item in state.tool_outputs)
    return total


@dataclass(frozen=True)
class ContextCompactionResult:
    triggered: bool
    reason: str
    previous_token_estimate: int
    compacted_token_estimate: int
    checkpoint: ContextCheckpoint | None = None


class ContextCompactor:
    def __init__(
        self,
        *,
        token_threshold: int = 1_500,
        target_token_threshold: int = 750,
        keep_recent_notes: int = 3,
        keep_recent_chunks: int = 3,
    ) -> None:
        self._token_threshold = max(token_threshold, 10)
        self._target_token_threshold = max(target_token_threshold, 5)
        self._keep_recent_notes = max(keep_recent_notes, 1)
        self._keep_recent_chunks = max(keep_recent_chunks, 1)

    def compact_if_needed(self, state: AgentState) -> ContextCompactionResult:
        previous = estimate_agent_state_tokens(state)
        if previous <= self._token_threshold:
            return ContextCompactionResult(
                triggered=False,
                reason="below_threshold",
                previous_token_estimate=previous,
                compacted_token_estimate=previous,
            )

        checkpoint = self._build_checkpoint(state=state, previous_token_estimate=previous)
        state.context_checkpoints.append(checkpoint.model_dump(mode="json"))

        compacted_note = f"context_compacted[{checkpoint.checkpoint_id}]: {checkpoint.summary[:240]}"
        preserved_notes = state.analysis_notes[-self._keep_recent_notes :] if state.analysis_notes else []
        preserved_notes = [note[:120] for note in preserved_notes]
        state.analysis_notes = [compacted_note, *preserved_notes]
        state.retrieved_context = state.retrieved_context[-self._keep_recent_chunks :]

        if state.tool_outputs:
            state.tool_outputs = state.tool_outputs[-1:]

        compacted = estimate_agent_state_tokens(state)
        return ContextCompactionResult(
            triggered=True,
            reason="token_threshold_exceeded",
            previous_token_estimate=previous,
            compacted_token_estimate=compacted,
            checkpoint=checkpoint,
        )

    def _build_checkpoint(self, *, state: AgentState, previous_token_estimate: int) -> ContextCheckpoint:
        citations = [
            f"{item.get('source', 'unknown')}#chunk-{item.get('chunk_id', 'n/a')}"
            for item in state.retrieved_context
            if item.get("source") is not None
        ][:8]
        open_tasks = [error for error in state.errors][-6:]
        tool_outputs = [str(item.get("tool", "tool")) + ":" + str(item.get("status", "unknown")) for item in state.tool_outputs][
            -6:
        ]
        pinned = PinnedContext(
            goals=[state.query],
            constraints=[
                f"allowed_tools={state.allowed_tools}",
                f"tool_budget_remaining={state.tool_budget}",
                f"max_steps={state.max_steps}",
            ],
            citations=citations,
            open_tasks=open_tasks,
            tool_outputs=tool_outputs,
        )
        summary_parts: list[str] = []
        if state.analysis_notes:
            summary_parts.append("notes=" + " | ".join(state.analysis_notes[-5:]))
        if state.retrieved_context:
            snippets = [str(item.get("snippet", "")).strip() for item in state.retrieved_context[-5:]]
            snippets = [snippet for snippet in snippets if snippet]
            if snippets:
                summary_parts.append("evidence=" + " | ".join(snippet[:160] for snippet in snippets))
        if state.tool_outputs:
            summary_parts.append("tools=" + " | ".join(tool_outputs))
        summary = " ; ".join(summary_parts).strip() or "Context compacted under token pressure."
        summary = summary[:360]
        return ContextCheckpoint(
            checkpoint_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            reason="token_threshold_exceeded",
            previous_token_estimate=previous_token_estimate,
            compacted_token_estimate=max(self._target_token_threshold, previous_token_estimate // 2),
            summary=summary,
            pinned_context=pinned,
            metadata={"strategy": "codex_style_summary_checkpoint"},
        )
