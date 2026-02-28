from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Trace(BaseModel):
    request_id: str
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class IngestRequest(BaseModel):
    sources: list[str] = Field(min_length=1)
    source_type: str = Field(default="mixed")


class IngestResponse(BaseModel):
    status: str
    accepted_sources: int
    trace: Trace


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    trace: Trace


class AgentRunRequest(BaseModel):
    query: str = Field(min_length=1)
    tools: list[str] | None = None


class AgentRunResponse(BaseModel):
    answer: str
    steps: list[str]
    tool_calls: list[str]
    confidence: float
    trace: Trace


class VisionRequest(BaseModel):
    image_uri: str = Field(min_length=1)
    prompt: str | None = None


class VisionResponse(BaseModel):
    summary: str
    findings: list[str]
    confidence: float
    trace: Trace


class VideoRequest(BaseModel):
    video_uri: str = Field(min_length=1)
    prompt: str | None = None
    sample_fps: float = Field(default=1.0, gt=0)
    max_frames: int = Field(default=32, ge=1, le=1000)


class VideoResponse(BaseModel):
    summary: str
    key_events: list[str]
    confidence: float
    processed_frames: int
    trace: Trace


class MetricsResponse(BaseModel):
    accuracy: float | None = None
    p95_latency_ms: int | None = None
    avg_cost_usd: float | None = None
    last_updated: datetime | None = None

