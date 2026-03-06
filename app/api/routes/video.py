import re

from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import Trace, VideoRequest, VideoResponse
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["video"])


@router.post("/video/analyze", response_model=VideoResponse)
async def analyze_video(
    payload: VideoRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> VideoResponse:
    summary = await container.video.analyze_video(
        video_uri=payload.video_uri,
        prompt=payload.prompt,
        sample_fps=payload.sample_fps,
        max_frames=payload.max_frames,
    )
    key_events = [part.strip(" -") for part in re.split(r"[;\n\.]+", summary) if part.strip()]
    if not key_events:
        key_events = [summary] if summary.strip() else ["No events available."]
    estimated_frames = max(1, min(payload.max_frames, int(payload.sample_fps * 8)))
    return VideoResponse(
        summary=summary,
        key_events=key_events[:5],
        confidence=0.6 if key_events else 0.2,
        processed_frames=estimated_frames,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
