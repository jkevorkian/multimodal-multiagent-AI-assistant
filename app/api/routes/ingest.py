from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import IngestRequest, IngestResponse, Trace
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["ingest"])


@router.post("/ingest/documents", response_model=IngestResponse)
async def ingest_documents(
    payload: IngestRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    _ = container
    return IngestResponse(
        status="accepted",
        accepted_sources=len(payload.sources),
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )

