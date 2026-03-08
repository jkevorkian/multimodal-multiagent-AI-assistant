from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.contracts.chat import (
    ChatFilesAttachRequest,
    ChatFilesAttachResponse,
    ChatFileRecord,
    ChatMessage,
    ChatMessagesResponse,
    ChatMessageSendRequest,
    ChatMessageSendResponse,
    ChatSession,
    ChatSessionCreateRequest,
    ChatSessionsResponse,
    ChatSessionUpdateRequest,
)
from app.contracts.schemas import Trace
from app.contracts.steering import SteeringApplied
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container
from app.core.steering import apply_answer_style, apply_tool_policy, enforce_grounding_policy, resolve_profile

router = APIRouter(tags=["chat"])


def _auto_mode(query: str, selected_tools: list[str] | None) -> str:
    if selected_tools:
        return "agentic"
    lowered = query.lower()
    keywords = ("latest", "search", "lookup", "tool", "api", "compare", "plan", "investigate")
    if any(token in lowered for token in keywords):
        return "agentic"
    return "rag"


def _looks_like_human_text(text: str) -> bool:
    if not text:
        return False
    length = len(text)
    printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    word_like = sum(1 for char in text if char.isalnum() or char.isspace() or char in ".,;:!?()[]{}'\"-_/@")
    alpha = sum(1 for char in text if char.isalpha())
    printable_ratio = printable / max(1, length)
    word_like_ratio = word_like / max(1, length)
    alpha_ratio = alpha / max(1, length)
    if length < 20:
        return alpha >= 1
    return printable_ratio >= 0.95 and word_like_ratio >= 0.65 and alpha_ratio >= 0.2


def _clean_snippet(snippet: str) -> str:
    compact = " ".join(snippet.split())
    if not _looks_like_human_text(compact):
        return ""
    return compact


def _fallback_answer_from_snippets(snippets: list[str]) -> str:
    if not snippets:
        return "The model returned an empty response and no readable evidence was available."
    excerpt = " ".join(snippets[:2]).strip()
    if len(excerpt) > 500:
        excerpt = excerpt[:500].rstrip() + "..."
    return f"The model returned an empty final response. Based on retrieved evidence: {excerpt}"


def _compose_chat_query(messages: list[dict[str, Any]], current_message: str, max_turns: int = 8) -> str:
    relevant = [item for item in messages if item.get("role") in {"user", "assistant"}][-max_turns:]
    lines: list[str] = []
    for item in relevant:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        role = str(item.get("role", "assistant")).upper()
        lines.append(f"{role}: {content}")
    transcript = "\n".join(lines)
    if not transcript:
        return current_message
    return (
        "Conversation context (most recent turns):\n"
        f"{transcript}\n\n"
        f"Current user request:\n{current_message}"
    )


def _format_sse_event(sequence_number: int, payload_json: str) -> str:
    return f"id: {sequence_number}\nevent: runtime.event\ndata: {payload_json}\n\n"


def _resolve_after_sequence(request: Request, after_sequence: int) -> int:
    if after_sequence > 0:
        return after_sequence
    last_event_id = request.headers.get("last-event-id", "").strip()
    if not last_event_id:
        return 0
    try:
        parsed = int(last_event_id)
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


def _metadata_filter_for_chat(*, chat_id: str, include_global_scope: bool) -> dict[str, Any] | None:
    if include_global_scope:
        return None
    return {"chat_id": chat_id, "scope": "chat"}


@router.post("/chat/sessions", response_model=ChatSession)
async def create_chat_session(
    payload: ChatSessionCreateRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatSession:
    session = container.chat_store.create_session(title=payload.title, defaults=payload.defaults)
    return ChatSession(**session)


@router.get("/chat/sessions", response_model=ChatSessionsResponse)
async def list_chat_sessions(
    request: Request,
    include_archived: bool = Query(default=False),
    container: ServiceContainer = Depends(get_container),
) -> ChatSessionsResponse:
    sessions = [ChatSession(**item) for item in container.chat_store.list_sessions(include_archived=include_archived)]
    return ChatSessionsResponse(
        sessions=sessions,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )


@router.get("/chat/sessions/{chat_id}", response_model=ChatSession)
async def get_chat_session(
    chat_id: str,
    container: ServiceContainer = Depends(get_container),
) -> ChatSession:
    session = container.chat_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")
    return ChatSession(**session)


@router.patch("/chat/sessions/{chat_id}", response_model=ChatSession)
async def patch_chat_session(
    chat_id: str,
    payload: ChatSessionUpdateRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatSession:
    session = container.chat_store.update_session(
        chat_id=chat_id,
        title=payload.title,
        archived=payload.archived,
        defaults=payload.defaults,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")
    return ChatSession(**session)


@router.get("/chat/sessions/{chat_id}/messages", response_model=ChatMessagesResponse)
async def list_chat_messages(
    chat_id: str,
    request: Request,
    limit: int = Query(default=500, ge=1, le=2000),
    container: ServiceContainer = Depends(get_container),
) -> ChatMessagesResponse:
    session = container.chat_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")
    messages = [ChatMessage(**item) for item in container.chat_store.list_messages(chat_id=chat_id, limit=limit)]
    files = [ChatFileRecord(**item) for item in container.chat_store.list_files(chat_id=chat_id, limit=limit)]
    return ChatMessagesResponse(
        chat_id=chat_id,
        messages=messages,
        files=files,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )


@router.post("/chat/sessions/{chat_id}/files", response_model=ChatFilesAttachResponse)
async def attach_chat_files(
    chat_id: str,
    payload: ChatFilesAttachRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> ChatFilesAttachResponse:
    session = container.chat_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")

    message_id = payload.message_id
    if not message_id:
        system_message = container.chat_store.add_message(
            chat_id=chat_id,
            role="system",
            content=f"Attached {len(payload.sources)} source(s).",
            mode="ingest",
            metadata={"kind": "file_attachment"},
        )
        message_id = system_message["message_id"]

    summary = await container.ingestion.ingest(
        payload.sources,
        payload.source_type,
        metadata_overrides={"chat_id": chat_id, "message_id": message_id, "scope": "chat"},
    )
    files = container.chat_store.add_files(
        chat_id=chat_id,
        message_id=message_id,
        sources=payload.sources,
        source_type=payload.source_type,
        metadata={"indexed_chunks": summary.indexed_chunks},
    )

    return ChatFilesAttachResponse(
        chat_id=chat_id,
        message_id=message_id,
        files=[ChatFileRecord(**item) for item in files],
        accepted_sources=summary.accepted_sources,
        indexed_chunks=summary.indexed_chunks,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )


@router.post("/chat/sessions/{chat_id}/messages", response_model=ChatMessageSendResponse)
async def send_chat_message(
    chat_id: str,
    payload: ChatMessageSendRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> ChatMessageSendResponse:
    session = container.chat_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")

    user_message = container.chat_store.add_message(
        chat_id=chat_id,
        role="user",
        content=payload.message,
        mode=payload.mode,
        metadata={"source_count": len(payload.sources)},
    )
    container.chat_store.touch_session(chat_id)

    accepted_sources = 0
    indexed_chunks = 0
    ingestion_status = "skipped"
    if payload.sources:
        ingest_metadata = {"chat_id": chat_id, "message_id": user_message["message_id"], "scope": "chat"}
        summary = await container.ingestion.ingest(
            payload.sources,
            payload.source_type,
            metadata_overrides=ingest_metadata,
        )
        accepted_sources = summary.accepted_sources
        indexed_chunks = summary.indexed_chunks
        ingestion_status = "accepted" if accepted_sources > 0 else "no_content"
        container.chat_store.add_files(
            chat_id=chat_id,
            message_id=user_message["message_id"],
            sources=payload.sources,
            source_type=payload.source_type,
            metadata={"indexed_chunks": indexed_chunks},
        )

    history = container.chat_store.list_messages(chat_id=chat_id, limit=40)
    composed_query = _compose_chat_query(history, payload.message)
    effective_mode = payload.mode if payload.mode != "auto" else _auto_mode(payload.message, payload.tools)
    metadata_filter = _metadata_filter_for_chat(chat_id=chat_id, include_global_scope=payload.include_global_scope)

    steering_resolution = resolve_profile(payload.steering)
    citations: list[str] = []
    confidence = 0.0
    run_id: str | None = None
    steering_notes: list[str] = [*steering_resolution.notes]
    answer = ""

    if effective_mode == "rag":
        try:
            context = await container.retriever.retrieve(
                composed_query,
                payload.top_k,
                metadata_filter=metadata_filter,
            )
        except Exception:
            context = []
        snippets = [cleaned for item in context if item.get("snippet") for cleaned in [_clean_snippet(item["snippet"])] if cleaned]
        try:
            answer = await container.llm.generate(composed_query, snippets)
        except Exception:
            answer = "The answer model failed for this request. Please try again."
        if not answer.strip():
            answer = _fallback_answer_from_snippets(snippets)
        citations = [f"{item['source']}#chunk-{item['chunk_id']}" for item in context]
        answer, grounding_notes = enforce_grounding_policy(
            answer=answer,
            citations=citations,
            profile=steering_resolution.profile,
            steering=payload.steering,
        )
        steering_notes.extend(grounding_notes)
        answer = apply_answer_style(answer, steering_resolution.profile)
        confidence = round(min(0.95, 0.35 + (0.12 * len(citations))), 2) if citations else 0.15
    else:
        run_id = payload.run_id or str(uuid4())
        available_tools = container.tool_registry.list_tools()
        enabled_tools, tool_policy_notes = apply_tool_policy(
            available_tools=available_tools,
            requested_tools=payload.tools,
            steering=payload.steering,
        )
        steering_notes.extend(tool_policy_notes)
        container.chat_store.upsert_run(
            chat_id=chat_id,
            message_id=user_message["message_id"],
            run_id=run_id,
            state="running",
        )
        orchestration_state = await container.orchestrator.run(
            query=composed_query,
            trace={"request_id": request.state.request_id, "trace_id": request.state.trace_id},
            run_id=run_id,
            allowed_tools=enabled_tools,
            tool_budget=settings.agent_tool_budget,
            max_steps=settings.agent_max_steps,
            resume_from_checkpoint=settings.agent_resume_from_checkpoint,
            retrieval_filter=metadata_filter,
        )
        citations = [
            f"{item['source']}#chunk-{item['chunk_id']}"
            for item in orchestration_state.retrieved_context
            if "source" in item and "chunk_id" in item
        ]
        grounded_answer, grounding_notes = enforce_grounding_policy(
            answer=orchestration_state.final_answer,
            citations=citations,
            profile=steering_resolution.profile,
            steering=payload.steering,
        )
        steering_notes.extend(grounding_notes)
        answer = apply_answer_style(grounded_answer, steering_resolution.profile)
        confidence = min(orchestration_state.confidence, 0.2) if grounded_answer != orchestration_state.final_answer else orchestration_state.confidence
        latest_status = await container.event_bus.latest_status(run_id)
        run_state = latest_status.state if latest_status is not None else "completed"
        container.chat_store.upsert_run(
            chat_id=chat_id,
            message_id=user_message["message_id"],
            run_id=run_id,
            state=run_state,
        )

    steering_applied = SteeringApplied(profile=steering_resolution.profile, notes=steering_notes)
    assistant_message = container.chat_store.add_message(
        chat_id=chat_id,
        role="assistant",
        content=answer,
        mode=effective_mode,
        run_id=run_id,
        metadata={
            "citations": citations,
            "confidence": confidence,
            "steering_applied": steering_applied.model_dump(mode="json"),
            "ingestion_status": ingestion_status,
            "accepted_sources": accepted_sources,
            "indexed_chunks": indexed_chunks,
        },
    )

    return ChatMessageSendResponse(
        chat_id=chat_id,
        user_message=ChatMessage(**user_message),
        assistant_message=ChatMessage(**assistant_message),
        mode=effective_mode if effective_mode in {"auto", "rag", "agentic"} else "rag",
        run_id=run_id,
        citations=citations,
        confidence=float(confidence),
        steering_applied=steering_applied,
        ingestion_status=ingestion_status,
        accepted_sources=accepted_sources,
        indexed_chunks=indexed_chunks,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )


@router.get("/chat/sessions/{chat_id}/runs/{run_id}/events")
async def stream_chat_run_events(
    chat_id: str,
    run_id: str,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    follow: bool = Query(default=False),
    container: ServiceContainer = Depends(get_container),
):
    session = container.chat_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat_not_found")
    if not container.chat_store.has_run(chat_id=chat_id, run_id=run_id):
        raise HTTPException(status_code=404, detail="chat_run_not_found")
    resolved_after_sequence = _resolve_after_sequence(request, after_sequence)

    async def event_generator():
        current_sequence = resolved_after_sequence
        buffered = await container.event_bus.list_events(run_id, after_sequence=current_sequence)
        for event in buffered:
            current_sequence = max(current_sequence, event.sequence_number)
            yield _format_sse_event(event.sequence_number, event.model_dump_json())
        if not follow:
            return

        while True:
            if await request.is_disconnected():
                return
            updates = await container.event_bus.wait_for_events(
                run_id,
                after_sequence=current_sequence,
                timeout_sec=15.0,
            )
            if not updates:
                yield ": keep-alive\n\n"
                continue
            for event in updates:
                current_sequence = max(current_sequence, event.sequence_number)
                yield _format_sse_event(event.sequence_number, event.model_dump_json())

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
