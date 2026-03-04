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
    snippets = [item["snippet"] for item in context if item.get("snippet")]
    answer = await container.llm.generate(payload.query, snippets)
    citations = [f"{item['source']}#chunk-{item['chunk_id']}" for item in context]
    confidence = round(min(0.95, 0.35 + (0.12 * len(citations))), 2) if citations else 0.15
    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
