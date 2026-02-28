from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if hasattr(record, "latency_ms"):
            payload["latency_ms"] = record.latency_ms
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(log_level: str) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(log_level.upper())


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))

        request.state.request_id = request_id
        request.state.trace_id = trace_id

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        response.headers["x-request-id"] = request_id
        response.headers["x-trace-id"] = trace_id

        logging.getLogger("http.request").info(
            "request_completed",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "latency_ms": latency_ms,
            },
        )
        return response

