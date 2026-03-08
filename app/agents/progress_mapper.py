from __future__ import annotations

from app.contracts.runtime_events import RunLifecycleState, RunStatusSnapshot, RuntimeEvent


class ProgressMapper:
    _STATE_BY_EVENT: dict[str, RunLifecycleState] = {
        "run.completed": "completed",
        "run.failed": "failed",
        "run.guardrail_triggered": "guardrail_exit",
    }

    def to_status_snapshot(self, event: RuntimeEvent) -> RunStatusSnapshot:
        state = self._STATE_BY_EVENT.get(event.event_type, "running")
        return RunStatusSnapshot(
            run_id=event.run_id,
            trace_id=event.trace_id,
            state=state,
            status_text=event.status_text,
            last_event_type=event.event_type,
            last_sequence_number=event.sequence_number,
            updated_at=event.timestamp,
            agent=event.agent,
            tool=event.tool,
            metadata=dict(event.metadata),
        )
