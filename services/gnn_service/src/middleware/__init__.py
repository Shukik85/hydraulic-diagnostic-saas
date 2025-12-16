"""Middleware package for FastAPI.

Provides:
- OpenTelemetry distributed tracing
- Rate limiting (token bucket)
- Request ID tracking (already in main.py)
- Body size limiting (already in main.py)
"""

from .opentelemetry import OpenTelemetryMiddleware, setup_opentelemetry
from .rate_limiter import RateLimitMiddleware

__all__ = [
    "OpenTelemetryMiddleware",
    "setup_opentelemetry",
    "RateLimitMiddleware",
]
