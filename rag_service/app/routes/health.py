"""Health check endpoints for RAG service."""

from fastapi import APIRouter, status
from pydantic import BaseModel
import time
import structlog

logger = structlog.get_logger()

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    service: str
    version: str


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint (no auth required for monitoring).
    
    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        service="rag-service",
        version="0.1.0"
    )


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check for Kubernetes/orchestration.
    
    Checks if service is ready to accept requests.
    """
    # TODO: Check if models are loaded, vector store is accessible, etc.
    return {"status": "ready"}


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """Liveness check for Kubernetes/orchestration.
    
    Simple check to see if service is alive.
    """
    return {"status": "alive"}
