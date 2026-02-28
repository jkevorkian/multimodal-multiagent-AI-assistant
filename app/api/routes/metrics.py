from datetime import datetime, timezone

from fastapi import APIRouter

from app.contracts.schemas import MetricsResponse

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(
        accuracy=None,
        p95_latency_ms=None,
        avg_cost_usd=None,
        last_updated=datetime.now(timezone.utc),
    )

