from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import RequestContextMiddleware, configure_logging


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logging.getLogger("app.startup").info(
            "service_started",
            extra={"component": "startup", "request_id": "system", "trace_id": "system"},
        )

    return app


app = create_app()

