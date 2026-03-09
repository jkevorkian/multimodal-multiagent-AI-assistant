from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.contracts.steering import SteeringApplied, SteeringRequest


class Trace(BaseModel):
    request_id: str
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class RuntimeModelEntry(BaseModel):
    subsystem: str
    component: str
    provider: str
    model: str | None = None
    details: list[str] = Field(default_factory=list)


class RuntimeModelsResponse(BaseModel):
    entries: list[RuntimeModelEntry]
    trace: Trace


class IngestRequest(BaseModel):
    sources: list[str] = Field(min_length=1)
    source_type: str = Field(default="mixed")
    metadata: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    status: str
    accepted_sources: int
    indexed_chunks: int = 0
    trace: Trace


class IndexedSourceInfo(BaseModel):
    source: str
    modality: str
    chunk_count: int
    min_chunk_id: int | None = None
    max_chunk_id: int | None = None
    min_offset: int | None = None
    max_offset: int | None = None
    sample_snippet: str | None = None


class IndexedSourcesResponse(BaseModel):
    count: int
    sources: list[IndexedSourceInfo]
    trace: Trace


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    steering: SteeringRequest | None = None
    metadata_filter: dict[str, Any] | None = None


class RetrievedChunk(BaseModel):
    source: str
    chunk_id: int = -1
    offset: int = -1
    snippet: str = ""
    score: float = 0.0
    modality: str = "text"
    timestamp_sec: float | None = None
    frame_index: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float
    steering_applied: SteeringApplied | None = None
    trace: Trace


class AgentRunRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    tools: list[str] | None = None
    run_id: str | None = None
    steering: SteeringRequest | None = None
    metadata_filter: dict[str, Any] | None = None


class AgentRunResponse(BaseModel):
    run_id: str
    answer: str
    steps: list[str]
    tool_calls: list[str]
    citations: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float
    steering_applied: SteeringApplied | None = None
    trace: Trace


class AgentToolInfo(BaseModel):
    name: str
    description: str


class AgentToolsResponse(BaseModel):
    tools: list[AgentToolInfo]


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
