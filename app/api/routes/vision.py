from fastapi import APIRouter, Depends, HTTPException, Request

from app.contracts.schemas import Trace, VisionRequest, VisionResponse
from app.core.config import settings
from app.core.dependencies import ServiceContainer, get_container
from app.vision.adapter import VisionAdapter, VisionAnalysis
from app.vision.fusion import VisionFusion
from app.vision.preprocess import VisionPreprocessor

router = APIRouter(tags=["vision"])


@router.post("/vision/analyze", response_model=VisionResponse)
async def analyze_image(
    payload: VisionRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> VisionResponse:
    preprocessor = VisionPreprocessor(max_source_bytes=settings.vision_max_source_bytes)
    adapter = VisionAdapter(vision_client=container.vision, preprocessor=preprocessor)
    fusion = VisionFusion()
    try:
        analysis = await adapter.analyze(payload.image_uri, payload.prompt)
        summary, findings, confidence = fusion.compose(analysis)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - provider/network edge
        try:
            processed = await preprocessor.preprocess(payload.image_uri)
        except ValueError as validation_exc:
            raise HTTPException(status_code=422, detail=str(validation_exc)) from validation_exc
        degraded = VisionAnalysis(
            processed_image=processed,
            prompt=payload.prompt,
            summary=(
                "Vision provider unavailable for this request. "
                "Returning degraded analysis from source metadata only."
            ),
        )
        summary, findings, confidence = fusion.compose(degraded)
        confidence = min(confidence, 0.2)
    return VisionResponse(
        summary=summary,
        findings=findings,
        confidence=confidence,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
