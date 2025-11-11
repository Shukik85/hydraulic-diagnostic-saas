"""
Quota and rate limiting middleware
Enforces subscription tier limits
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog
import redis.asyncio as redis

from ..config import settings

logger = structlog.get_logger()


class QuotaMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting based on subscription tier
    Uses Redis for distributed counting
    """

    def __init__(self, app):
        super().__init__(app)
        self.redis_client = None

    async def dispatch(self, request: Request, call_next):
        # Skip health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Get user from request state (set by AuthMiddleware)
        api_key = request.headers.get(settings.API_KEY_HEADER)

        if api_key:
            # Check rate limit
            is_allowed = await self._check_rate_limit(api_key)

            if not is_allowed:
                logger.warning("rate_limit_exceeded", api_key_prefix=api_key[:10])
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Upgrade your plan."}
                )

        response = await call_next(request)
        return response

    async def _check_rate_limit(self, api_key: str) -> bool:
        """
        Check if user has quota remaining
        TODO: Implement Redis-based rate limiting
        """
        # Placeholder: always allow for now
        return True

    async def _get_redis_client(self):
        """Get or create Redis connection"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
        return self.redis_client
