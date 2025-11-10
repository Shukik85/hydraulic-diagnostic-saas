"""Internal API key authentication for RAG Service."""
from fastapi import Header, HTTPException, status
from app.config import get_settings
settings = get_settings()
async def verify_api_key(x_internal_api_key: str = Header(...)) -> str:
    if x_internal_api_key != settings.rag_internal_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key", headers={"WWW-Authenticate": "ApiKey"})
    return x_internal_api_key
