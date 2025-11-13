"""Production middleware stack for FastAPI."""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
import uuid
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """Global exception handler with trace_id correlation."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} {response.status_code}",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration,
                }
            )
            return response
            
        except Exception as exc:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {str(exc)}",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration,
                    "exception": type(exc).__name__,
                }
            )
            
            from app.exceptions import AppException, ErrorCode
            
            if isinstance(exc, AppException):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": exc.code.value,
                        "message": exc.message,
                        "trace_id": request_id,
                    },
                )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": ErrorCode.INTERNAL_ERROR.value,
                    "message": "Internal server error",
                    "trace_id": request_id,
                },
            )
