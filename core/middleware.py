"""
Enterprise middleware for RAG_MONSTER.

Starlette BaseHTTPMiddleware subclasses:
  - RequestIDMiddleware  — injects X-Request-ID into request & response
  - APIKeyMiddleware     — enforces API_KEY (mandatory when HOST != 127.0.0.1) using hmac.compare_digest
  - RateLimitMiddleware  — Redis sliding-window limiter with in-memory fallback & proxy security
"""

from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from collections import deque
from typing import Callable

try:
    import structlog
    _HAS_STRUCTLOG = True
    logger = structlog.get_logger(__name__)
except ImportError:
    _HAS_STRUCTLOG = False
    logger = logging.getLogger(__name__)

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.types import ASGIApp
    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False
    class BaseHTTPMiddleware:
        def __init__(self, app=None):
            self.app = app
    Request = object
    Response = object
    JSONResponse = object
    ASGIApp = object


# ---------------------------------------------------------------------------
# Paths that bypass auth checks (Notice /metrics is NOT exempt!)
# ---------------------------------------------------------------------------
_AUTH_SKIP_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/v1/health",
        "/v1/ready",
    }
)


# ---------------------------------------------------------------------------
# RequestIDMiddleware
# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a unique X-Request-ID header into every request and response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        if _HAS_STRUCTLOG:
            structlog.contextvars.bind_contextvars(request_id=request_id)

        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)
        finally:
            if _HAS_STRUCTLOG:
                structlog.contextvars.unbind_contextvars("request_id")

        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# APIKeyMiddleware
# ---------------------------------------------------------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Security enforcement:
      - Mandatory when HOST is not 127.0.0.1 / localhost.
      - Uses constant-time hmac.compare_digest to prevent timing attacks.
      - /metrics is protected alongside data endpoints.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._api_key: str = os.getenv("API_KEY", "").strip()
        self._host: str = os.getenv("HOST", "0.0.0.0").strip()

        # Enforce security configuration: non-localhost bindings MUST have an API_KEY set
        self._is_local_only: bool = self._host in ("127.0.0.1", "localhost")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # If bound to external interface without API_KEY set, block access
        if not self._is_local_only and not self._api_key:
            if request.url.path not in _AUTH_SKIP_PATHS and not request.url.path.startswith(("/docs", "/redoc")):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "Security policy: API_KEY must be configured when server binds "
                            f"to non-localhost interface ({self._host})."
                        )
                    },
                )

        # In local open mode (127.0.0.1 without key), allow open access
        if self._is_local_only and not self._api_key:
            return await call_next(request)

        # Allow health / docs paths
        if request.url.path in _AUTH_SKIP_PATHS or request.url.path.startswith(("/docs", "/redoc")):
            return await call_next(request)

        auth_header: str = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or malformed Authorization header."},
            )

        provided_key = auth_header[len("Bearer "):]
        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(provided_key.encode("utf-8"), self._api_key.encode("utf-8")):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter with Redis backend and in-memory fallback.
    Ignores X-Forwarded-For unless client is in TRUSTED_PROXIES.
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(self, app: ASGIApp, redis_client=None) -> None:
        super().__init__(app)
        self._limit: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
        self.redis_client = redis_client
        self._windows: dict[str, deque[float]] = {}

        # Parse trusted proxies
        trusted_raw = os.getenv("TRUSTED_PROXIES", "")
        self._trusted_proxies: set[str] = {
            p.strip() for p in trusted_raw.split(",") if p.strip()
        }

    def _get_client_ip(self, request: Request) -> str:
        peer_ip = request.client.host if request.client else "unknown"

        # Only honor X-Forwarded-For if request comes directly from a trusted reverse proxy
        if peer_ip in self._trusted_proxies and "X-Forwarded-For" in request.headers:
            return request.headers["X-Forwarded-For"].split(",")[0].strip()

        return peer_ip

    def _is_allowed(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - self._WINDOW_SECONDS

        # 1. Try Redis sliding window via sorted set
        if self.redis_client:
            try:
                key = f"rate_limit:{ip}"
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, cutoff)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, int(self._WINDOW_SECONDS) + 5)
                res = pipe.execute()
                current_count = res[1]
                if current_count >= self._limit:
                    return False
                return True
            except Exception as exc:
                logger.warning("rate_limit_redis_error", extra={"error": str(exc)})

        # 2. In-memory sliding window fallback
        window = self._windows.setdefault(ip, deque())
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._limit:
            return False

        window.append(now)
        return True

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._limit <= 0:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if not self._is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self._limit} requests per minute."
                },
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
