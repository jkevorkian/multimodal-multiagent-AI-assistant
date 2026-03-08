from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RuntimeEventType = Literal[
    "run.started",
    "agent.step.started",
    "agent.step.completed",
    "tool.call.started",
    "tool.call.completed",
    "tool.call.failed",
    "model.call.in_progress",
    "agent.revision.requested",
    "run.completed",
    "run.failed",
    "run.guardrail_triggered",
]

RunLifecycleState = Literal["running", "completed", "failed", "guardrail_exit"]


class RuntimeEvent(BaseModel):
    run_id: str
    trace_id: str
    sequence_number: int = Field(ge=1)
    timestamp: datetime
    event_type: RuntimeEventType
    status_text: str
    agent: str | None = None
    tool: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStatusSnapshot(BaseModel):
    run_id: str
    trace_id: str
    state: RunLifecycleState
    status_text: str
    last_event_type: RuntimeEventType
    last_sequence_number: int = Field(ge=1)
    updated_at: datetime
    agent: str | None = None
    tool: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
