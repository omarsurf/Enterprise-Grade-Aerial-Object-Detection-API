"""
Compatibility wrapper around slowapi with a lightweight local fallback.
"""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from functools import wraps

from fastapi import Request

try:  # pragma: no cover - exercised when slowapi is installed
    from slowapi import Limiter  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
except ImportError:

    class RateLimitExceeded(Exception):  # noqa: N818
        """Fallback rate-limit exception compatible with the app handler."""

        def __init__(self, detail: str):
            super().__init__(detail)
            self.detail = detail

    class _InMemoryStorage:
        def __init__(self):
            self._buckets = defaultdict(list)

        def reset(self) -> None:
            self._buckets.clear()

        def clear(self) -> None:
            self.reset()

    class Limiter:
        """Small in-memory limiter used only when slowapi is unavailable."""

        def __init__(self, key_func, default_limits=None, headers_enabled=False):
            self.key_func = key_func
            self.default_limits = default_limits or []
            self.headers_enabled = headers_enabled
            self._storage = _InMemoryStorage()

        def limit(self, limit_value: str):
            max_requests, window_seconds = _parse_limit(limit_value)

            def decorator(func):
                signature = inspect.signature(func)

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    request = _find_request_arg(args, kwargs)
                    bucket_key = (func.__name__, self.key_func(request), window_seconds)
                    bucket = self._storage._buckets[bucket_key]
                    now = time.time()
                    cutoff = now - window_seconds
                    while bucket and bucket[0] <= cutoff:
                        bucket.pop(0)

                    if len(bucket) >= max_requests:
                        raise RateLimitExceeded(limit_value)

                    bucket.append(now)
                    return await func(*args, **kwargs)

                wrapper.__signature__ = signature
                return wrapper

            return decorator


def _parse_limit(limit_value: str) -> tuple[int, int]:
    raw_count, raw_window = limit_value.split("/", 1)
    count = int(raw_count.strip())
    window_key = raw_window.strip().lower()

    if window_key in {"s", "sec", "second", "seconds"}:
        return count, 1
    if window_key in {"m", "min", "minute", "minutes"}:
        return count, 60
    if window_key in {"h", "hour", "hours"}:
        return count, 3600
    raise ValueError(f"Unsupported limit window: {limit_value}")


def _find_request_arg(args, kwargs) -> Request:
    for value in list(kwargs.values()) + list(args):
        if isinstance(value, Request):
            return value
    raise RuntimeError("Rate-limited endpoints must receive a Request argument")
