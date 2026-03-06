from fastapi import APIRouter, Depends, Request

from app.contracts.schemas import QueryRequest, QueryResponse, Trace
from app.core.dependencies import ServiceContainer, get_container

router = APIRouter(tags=["query"])


def _looks_like_human_text(text: str) -> bool:
    if not text:
        return False
    length = len(text)
    printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    word_like = sum(
        1 for char in text if char.isalnum() or char.isspace() or char in ".,;:!?()[]{}'\"-_/@"
    )
    alpha = sum(1 for char in text if char.isalpha())
    printable_ratio = printable / max(1, length)
    word_like_ratio = word_like / max(1, length)
    alpha_ratio = alpha / max(1, length)
    if length < 20:
        return alpha >= 1
    return printable_ratio >= 0.95 and word_like_ratio >= 0.65 and alpha_ratio >= 0.2


def _clean_snippet(snippet: str) -> str:
    compact = " ".join(snippet.split())
    if not _looks_like_human_text(compact):
        return ""
    return compact


def _fallback_answer_from_snippets(snippets: list[str]) -> str:
    if not snippets:
        return "The model returned an empty response and no readable evidence was available."
    excerpt = " ".join(snippets[:2]).strip()
    if len(excerpt) > 500:
        excerpt = excerpt[:500].rstrip() + "..."
    return (
        "The model returned an empty final response. "
        f"Based on retrieved evidence: {excerpt}"
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    try:
        context = await container.retriever.retrieve(payload.query, payload.top_k)
    except Exception:
        context = []
    snippets = [cleaned for item in context if item.get("snippet") for cleaned in [_clean_snippet(item["snippet"])] if cleaned]
    try:
        answer = await container.llm.generate(payload.query, snippets)
    except Exception as exc:
        lowered = str(exc).lower()
        if "insufficient_quota" in lowered or "429" in lowered or "rate limit" in lowered:
            answer = (
                "The configured LLM provider is currently rate-limited or out of quota. "
                "Try again later or switch to heuristic fallback mode."
            )
        else:
            answer = "The answer model failed for this request. Please try again."
    if not answer.strip():
        answer = _fallback_answer_from_snippets(snippets)
    citations = [f"{item['source']}#chunk-{item['chunk_id']}" for item in context]
    confidence = round(min(0.95, 0.35 + (0.12 * len(citations))), 2) if citations else 0.15
    if (
        "rate-limited or out of quota" in answer
        or "failed for this request" in answer
        or "returned an empty final response" in answer
        or "returned an empty response" in answer
    ):
        confidence = min(confidence, 0.2)
    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        trace=Trace(request_id=request.state.request_id, trace_id=request.state.trace_id),
    )
