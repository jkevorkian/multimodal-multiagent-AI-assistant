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
    ingestion_summary = await container.ingestion.ingest(payload.sources, payload.source_type)
    return IngestResponse(
        status="accepted" if ingestion_summary.accepted_sources > 0 else "no_content",
        accepted_sources=ingestion_summary.accepted_sources,
        indexed_chunks=ingestion_summary.indexed_chunks,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
