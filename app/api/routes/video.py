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
    return VideoResponse(
        summary=summary,
        key_events=["stub_event"],
        confidence=0.2,
        processed_frames=min(payload.max_frames, 8),
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )

