"""
Authentication and rate-limit key helpers for the API surface.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import API_KEY_HEADER_NAME, DEFAULT_DEV_API_KEY, logger

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    """Authentication metadata safe to keep on the request state."""

    api_key_fingerprint: str


def fingerprint_api_key(api_key: str) -> str:
    """Builds a stable, non-reversible fingerprint for logs and rate limiting."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _parse_api_keys(raw_value: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in raw_value.split(",") if key.strip())


@lru_cache
def get_allowed_api_keys() -> tuple[str, ...]:
    """
    Returns configured API keys.

    A local development fallback keeps the sample app runnable without extra setup.
    """
    raw_value = os.getenv("AERIAL_API_KEYS", "").strip()
    if raw_value:
        keys = _parse_api_keys(raw_value)
        if keys:
            return keys

    fallback_fingerprint = fingerprint_api_key(DEFAULT_DEV_API_KEY)
    logger.warning(
        "AERIAL_API_KEYS is not set; using the local development API key fingerprint=%s",
        fallback_fingerprint,
    )
    return (DEFAULT_DEV_API_KEY,)


def clear_api_key_cache() -> None:
    """Test helper for refreshing env-driven key configuration."""
    get_allowed_api_keys.cache_clear()


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def get_rate_limit_key(request: Request) -> str:
    """Builds the in-memory rate-limit key without persisting the raw API key."""
    api_key = request.headers.get(API_KEY_HEADER_NAME)
    if api_key:
        return f"api-key:{fingerprint_api_key(api_key)}"
    return f"ip:{_get_client_ip(request)}"


def authenticate_api_key(api_key: str | None) -> AuthContext:
    """Validates the presented API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER_NAME} header.",
        )

    allowed_keys = get_allowed_api_keys()
    if api_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return AuthContext(api_key_fingerprint=fingerprint_api_key(api_key))


async def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> AuthContext:
    """FastAPI dependency that authenticates and annotates the request."""
    auth_context = authenticate_api_key(api_key)
    request.state.api_key_fingerprint = auth_context.api_key_fingerprint
    return auth_context
