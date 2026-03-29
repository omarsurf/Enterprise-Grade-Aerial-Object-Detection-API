"""
Request ID propagation and structured request logging.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response

from app.config import REQUEST_ID_HEADER_NAME, logger

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def build_request_id(incoming_request_id: str | None) -> str:
    """Returns a safe request id for correlation across logs and responses."""
    if incoming_request_id and REQUEST_ID_PATTERN.fullmatch(incoming_request_id):
        return incoming_request_id
    return uuid.uuid4().hex


def get_client_ip(request: Request) -> str:
    """Returns the best-effort client IP address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def log_request_event(request: Request, response: Response, duration_ms: float) -> None:
    """Emits a structured access log without leaking the raw API key."""
    payload = {
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "client_ip": get_client_ip(request),
        "api_key_fingerprint": getattr(request.state, "api_key_fingerprint", None),
        "model_version": getattr(request.app.state, "model_version", None),
    }
    logger.info(json.dumps(payload, sort_keys=True))


def install_observability(app: FastAPI) -> None:
    """Registers request-id propagation and structured logging middleware."""

    @app.middleware("http")
    async def request_observability(request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        request.state.request_id = build_request_id(request.headers.get(REQUEST_ID_HEADER_NAME))
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER_NAME] = request.state.request_id
        log_request_event(request, response, (time.perf_counter() - start_time) * 1000)
        return response
