"""Internal API key authentication."""

from fastapi import Header, HTTPException, status

from app.config import get_settings

settings = get_settings()


async def verify_api_key(x_internal_api_key: str = Header(...)) -> str:
    """Verify internal API key.
    
    Args:
        x_internal_api_key: API key from header
    
    Returns:
        Validated API key
    
    Raises:
        HTTPException: If key is invalid
    """
    if x_internal_api_key != settings.gnn_internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return x_internal_api_key
