from __future__ import annotations

from pydantic import BaseModel, Field


class GroundingPolicy(BaseModel):
    min_citations: int = Field(default=0, ge=0, le=10)
    abstain_on_insufficient_citations: bool = False


class ToolPolicy(BaseModel):
    allow: list[str] | None = None
    deny: list[str] | None = None
    require: list[str] | None = None


class SteeringRequest(BaseModel):
    profile: str = Field(default="balanced")
    grounding: GroundingPolicy | None = None
    tools: ToolPolicy | None = None


class SteeringApplied(BaseModel):
    profile: str
    notes: list[str] = Field(default_factory=list)
