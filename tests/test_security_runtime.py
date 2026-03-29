"""
Tests for API key handling and observability helpers.
"""

from fastapi import Request

from app.observability import build_request_id
from app.security import (
    authenticate_api_key,
    clear_api_key_cache,
    fingerprint_api_key,
    get_allowed_api_keys,
    get_rate_limit_key,
)


def build_request(headers: dict[str, str]) -> Request:
    """Constructs a minimal Starlette request for helper tests."""
    encoded_headers = [(key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/predict",
        "headers": encoded_headers,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def test_authenticate_api_key_uses_dev_fallback(monkeypatch):
    """A local fallback key keeps the app runnable without extra env setup."""
    monkeypatch.delenv("AERIAL_API_KEYS", raising=False)
    clear_api_key_cache()

    context = authenticate_api_key("local-dev-key")

    assert context.api_key_fingerprint == fingerprint_api_key("local-dev-key")
    assert get_allowed_api_keys() == ("local-dev-key",)


def test_get_rate_limit_key_uses_api_key_fingerprint():
    """Rate limiting should never store the raw API key."""
    request = build_request({"X-API-Key": "secret-key"})

    assert get_rate_limit_key(request) == f"api-key:{fingerprint_api_key('secret-key')}"


def test_build_request_id_accepts_safe_values_and_replaces_invalid():
    """Request IDs should accept safe incoming values and sanitize invalid ones."""
    assert build_request_id("req-123") == "req-123"
    assert build_request_id("bad value with spaces") != "bad value with spaces"
