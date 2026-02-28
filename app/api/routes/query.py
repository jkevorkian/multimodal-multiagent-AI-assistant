from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import QueryRequest, QueryResponse, Trace
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    context = await container.retriever.retrieve(payload.query, payload.top_k)
    answer = await container.llm.generate(payload.query, [c["snippet"] for c in context])
    return QueryResponse(
        answer=answer,
        citations=[c["source"] for c in context],
        confidence=0.25,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )

