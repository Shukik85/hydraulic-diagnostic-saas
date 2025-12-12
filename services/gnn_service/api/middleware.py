"""FastAPI middleware.

Custom middleware for:
- Request ID tracking
- Correlation logging
- Performance monitoring

Python 3.14 Features:
    - Deferred annotations
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID tracking to all requests.
    
    Adds X-Request-ID header to requests/responses for correlation.
    If client provides X-Request-ID, use it; otherwise generate UUID.
    
    Examples:
        Request without ID:
            GET /health
            -> X-Request-ID: auto-generated-uuid
        
        Request with ID:
            GET /health
            X-Request-ID: client-provided-id
            -> X-Request-ID: client-provided-id (same)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with ID tracking.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
        
        Returns:
            response: Response with X-Request-ID header
        """
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")

        if not request_id:
            request_id = str(uuid.uuid4())

        # Add to request state
        request.state.request_id = request_id

        # Log request start
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - Start",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path}
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log request completion
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Complete ({response.status_code}) - {duration_ms:.2f}ms",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms
            }
        )

        return response
