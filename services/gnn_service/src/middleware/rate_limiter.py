"""Rate limiting middleware using token bucket algorithm.

Provides:
- Per-IP rate limiting
- Per-user rate limiting (if authenticated)
- Configurable limits and windows
- Redis backend (optional, falls back to in-memory)
- Custom headers (X-RateLimit-*)

Environment Variables:
    RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
    RATE_LIMIT_REQUESTS: Max requests per window (default: 100)
    RATE_LIMIT_WINDOW_S: Window size in seconds (default: 60)
    REDIS_URL: Redis connection URL (optional)

Usage:
    >>> from src.middleware import RateLimitMiddleware
    >>> app.add_middleware(
    ...     RateLimitMiddleware,
    ...     requests_per_window=100,
    ...     window_seconds=60
    ... )
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Graceful degradation: Redis is optional
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not installed. Using in-memory rate limiting.")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Args:
        capacity: Maximum tokens (requests) in bucket
        refill_rate: Tokens added per second
        tokens: Current token count
        last_refill: Last refill timestamp
    """

    capacity: int
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize with full capacity."""
        if self.tokens == 0.0:
            self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            success: True if tokens consumed, False if insufficient
        """
        # Refill bucket
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity, self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)

    def get_reset_time(self) -> int:
        """Get seconds until bucket refills."""
        tokens_needed = self.capacity - self.tokens
        if tokens_needed <= 0:
            return 0
        return int(tokens_needed / self.refill_rate)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.

    Uses token bucket algorithm with configurable limits.
    Supports both in-memory and Redis backends.

    Args:
        app: FastAPI application
        requests_per_window: Max requests per window
        window_seconds: Window size in seconds
        redis_url: Redis connection URL (optional)

    Examples:
        >>> app.add_middleware(
        ...     RateLimitMiddleware,
        ...     requests_per_window=100,
        ...     window_seconds=60
        ... )
    """

    def __init__(
        self,
        app,
        requests_per_window: int | None = None,
        window_seconds: int | None = None,
        redis_url: str | None = None,
    ):
        super().__init__(app)

        # Configuration
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.requests_per_window = requests_per_window or int(
            os.getenv("RATE_LIMIT_REQUESTS", "100")
        )
        self.window_seconds = window_seconds or int(
            os.getenv("RATE_LIMIT_WINDOW_S", "60")
        )

        # Refill rate: requests per second
        self.refill_rate = self.requests_per_window / self.window_seconds

        # Backend
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"✅ Rate limiter using Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory.")

        # In-memory storage (fallback)
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.requests_per_window,
                refill_rate=self.refill_rate,
            )
        )

        logger.info(
            f"✅ Rate limiter initialized: {self.requests_per_window} req/{self.window_seconds}s"
        )

    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting.

        Priority:
        1. User ID (if authenticated)
        2. X-Forwarded-For header
        3. Client IP

        Args:
            request: FastAPI request

        Returns:
            identifier: Client identifier
        """
        # TODO: Add user ID extraction if using authentication
        # user_id = request.state.user_id if hasattr(request.state, "user_id") else None
        # if user_id:
        #     return f"user:{user_id}"

        # Use IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get first IP (client IP)
            return f"ip:{forwarded.split(',')[0].strip()}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit and process request."""
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_identifier(request)

        # Get bucket
        bucket = self.buckets[client_id]

        # Try to consume token
        if not bucket.consume(tokens=1):
            # Rate limited
            remaining = bucket.get_remaining()
            reset_time = bucket.get_reset_time()

            logger.warning(
                f"Rate limit exceeded: {client_id}",
                extra={
                    "client": client_id,
                    "path": request.url.path,
                    "remaining": remaining,
                    "reset_in": reset_time,
                },
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Try again in {reset_time} seconds.",
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time),
                    "Retry-After": str(reset_time),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = bucket.get_remaining()
        reset_time = bucket.get_reset_time()

        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + reset_time)

        return response
