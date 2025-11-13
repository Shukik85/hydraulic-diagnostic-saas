"""Global exception handling for production-grade API."""
from typing import Any
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    DATABASE_ERROR = "DATABASE_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT = "TIMEOUT"

class AppException(Exception):
    """Base exception for application-level errors."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(message)

class ValidationError(AppException):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=422,
            details=details,
        )

class InferenceError(AppException):
    """Raised when model inference fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.INFERENCE_FAILED,
            message=message,
            status_code=503,
            details=details,
        )

class DatabaseError(AppException):
    """Raised when database operation fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            status_code=503,
            details=details,
        )
