"""Internal authentication for ML service."""

from fastapi import Header, HTTPException, status
import structlog

from config import settings

logger = structlog.get_logger()


async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., description="Internal API key for backend→ml_service auth")
) -> bool:
    """Verify internal API key for backend→ml_service communication.
    
    Args:
        x_internal_api_key: API key from request header
    
    Returns:
        True if valid
    
    Raises:
        HTTPException: If API key is invalid (403 Forbidden)
    """
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
