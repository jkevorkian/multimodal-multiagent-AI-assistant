from fastapi import APIRouter

from app.api.routes import agents, health, ingest, metrics, query, runs, video, vision

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ingest.router)
api_router.include_router(query.router)
api_router.include_router(agents.router)
api_router.include_router(runs.router)
api_router.include_router(vision.router)
api_router.include_router(video.router)
api_router.include_router(metrics.router)
