"""
Health check endpoints for monitoring and orchestration
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from ..db.session import get_db

router = APIRouter()
logger = structlog.get_logger()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness probe (checks DB connectivity)"""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error("readiness_check_failed", exc_info=e)
        return {"status": "not_ready", "database": "error"}


@router.get("/live")
async def liveness_check():
    """Liveness probe"""
    return {"status": "alive"}
