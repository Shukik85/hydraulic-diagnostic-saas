"""Internal authentication for RAG service."""

from fastapi import Header, HTTPException, status
from .config import get_settings
import structlog

logger = structlog.get_logger()


async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., description="Internal API key for service-to-service auth")
) -> bool:
    """Verify internal API key for backend→rag_service communication.
    
    Args:
        x_internal_api_key: API key from request header
    
    Returns:
        True if valid
    
    Raises:
        HTTPException: If API key is invalid (403 Forbidden)
    """
    settings = get_settings()
    
    if x_internal_api_key != settings.internal_api_key:
        logger.warning(
            "Invalid internal API key attempt",
            provided_key=x_internal_api_key[:8] + "***" if x_internal_api_key else None
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key"
        )
    
    return True
