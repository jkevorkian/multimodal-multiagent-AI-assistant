from fastapi import APIRouter

from app.contracts.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)

