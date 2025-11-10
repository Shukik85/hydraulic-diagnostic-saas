"""Internal authentication for RAG service."""

from fastapi import Header, HTTPException, status
import structlog

from app.config import settings

logger = structlog.get_logger()

async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., description="Internal API key for backend→rag_service auth")
) -> bool:
    """Verify internal API key for backend→rag_service communication."""
    if x_internal_api_key != settings.internal_api_key:
        logger.warning("Invalid internal API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key"
        )
    return True
