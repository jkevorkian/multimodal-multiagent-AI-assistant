from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.schemas import Trace
from app.contracts.steering import SteeringApplied, SteeringRequest

ChatMode = Literal["auto", "rag", "agentic"]


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None
    defaults: dict[str, Any] | None = None


class ChatSessionUpdateRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None
    defaults: dict[str, Any] | None = None


class ChatSession(BaseModel):
    chat_id: str
    title: str
    archived: bool = False
    defaults: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSession]
    trace: Trace


class ChatMessage(BaseModel):
    message_id: str
    chat_id: str
    role: str
    content: str
    mode: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ChatFileRecord(BaseModel):
    file_id: str
    chat_id: str
    message_id: str
    uri: str
    source_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ChatMessagesResponse(BaseModel):
    chat_id: str
    messages: list[ChatMessage]
    files: list[ChatFileRecord] = Field(default_factory=list)
    trace: Trace


class ChatMessageSendRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: ChatMode = "auto"
    top_k: int = Field(default=5, ge=1, le=20)
    tools: list[str] | None = None
    steering: SteeringRequest | None = None
    source_type: str = "mixed"
    sources: list[str] = Field(default_factory=list)
    include_global_scope: bool = False
    run_id: str | None = None


class ChatFilesAttachRequest(BaseModel):
    sources: list[str] = Field(min_length=1)
    source_type: str = "mixed"
    message_id: str | None = None


class ChatFilesAttachResponse(BaseModel):
    chat_id: str
    message_id: str
    files: list[ChatFileRecord] = Field(default_factory=list)
    accepted_sources: int = 0
    indexed_chunks: int = 0
    trace: Trace


class ChatMessageSendResponse(BaseModel):
    chat_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    mode: ChatMode
    run_id: str | None = None
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    steering_applied: SteeringApplied | None = None
    ingestion_status: str = "skipped"
    accepted_sources: int = 0
    indexed_chunks: int = 0
    trace: Trace
