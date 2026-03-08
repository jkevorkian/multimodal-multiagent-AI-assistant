from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PinnedContext(BaseModel):
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    open_tasks: list[str] = Field(default_factory=list)
    tool_outputs: list[str] = Field(default_factory=list)


class ContextCheckpoint(BaseModel):
    checkpoint_id: str
    created_at: datetime
    reason: str
    previous_token_estimate: int = Field(ge=0)
    compacted_token_estimate: int = Field(ge=0)
    summary: str
    pinned_context: PinnedContext
    metadata: dict[str, Any] = Field(default_factory=dict)
