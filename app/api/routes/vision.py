from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import Trace, VisionRequest, VisionResponse
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["vision"])


@router.post("/vision/analyze", response_model=VisionResponse)
async def analyze_image(
    payload: VisionRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> VisionResponse:
    summary = await container.vision.analyze_image(payload.image_uri, payload.prompt)
    return VisionResponse(
        summary=summary,
        findings=["stub_finding"],
        confidence=0.2,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )

