from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["runs"])


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


@router.get("/runs/{run_id}/status")
async def get_run_status(
    run_id: str,
    container: ServiceContainer = Depends(get_container),
):
    status = await container.event_bus.latest_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return status


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    follow: bool = Query(default=False),
    container: ServiceContainer = Depends(get_container),
):
    resolved_after_sequence = _resolve_after_sequence(request, after_sequence)
    run_exists = await container.event_bus.has_run(run_id)
    if not run_exists and not follow:
        raise HTTPException(status_code=404, detail="run_not_found")

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
