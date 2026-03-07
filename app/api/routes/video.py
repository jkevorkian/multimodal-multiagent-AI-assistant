from fastapi import APIRouter, Depends, HTTPException, Request

from app.contracts.schemas import Trace, VideoRequest, VideoResponse
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container
from app.video import TemporalAggregator, VideoAnalysisAdapter, VideoFrameSampler

router = APIRouter(tags=["video"])


@router.post("/video/analyze", response_model=VideoResponse)
async def analyze_video(
    payload: VideoRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> VideoResponse:
    adapter = VideoAnalysisAdapter(
        video_client=container.video,
        vision_client=container.vision,
        require_frame_level_findings=settings.multimodal_video_require_frame_findings,
        frame_sampler=VideoFrameSampler(
            latency_budget_ms=settings.multimodal_video_latency_budget_ms,
            estimated_ms_per_frame=settings.multimodal_video_estimated_ms_per_frame,
            max_window_sec=settings.multimodal_video_max_window_sec,
            remote_fetch_timeout_sec=settings.multimodal_video_remote_fetch_timeout_sec,
            max_remote_source_bytes=settings.multimodal_video_max_remote_source_bytes,
        ),
        temporal_aggregator=TemporalAggregator(max_key_events=settings.multimodal_video_max_key_events),
    )
    try:
        analysis = await adapter.analyze(
            video_uri=payload.video_uri,
            prompt=payload.prompt,
            sample_fps=payload.sample_fps,
            max_frames=payload.max_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"video_analysis_unavailable:{exc}") from exc

    return VideoResponse(
        summary=analysis.summary,
        key_events=analysis.key_events,
        confidence=analysis.confidence,
        processed_frames=analysis.processed_frames,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
