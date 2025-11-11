"""
Authentication middleware
API key validation and user extraction
"""
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from db.session import get_db
from models.user import User
from config import settings

logger = structlog.get_logger()

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def get_current_user(
    api_key: str = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to extract current user from API key
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    # Query user by API key
    result = await db.execute(
        select(User).where(User.api_key == api_key)
    )
    user = result.scalar_one_or_none()

    if not user:
        logger.warning("invalid_api_key_attempt", api_key_prefix=api_key[:10])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    if user.subscription_status not in ["active", "trial"]:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Subscription expired or cancelled"
        )

    return user


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging authentication attempts
    """
    async def dispatch(self, request: Request, call_next):
        # Extract API key from header
        api_key = request.headers.get(settings.API_KEY_HEADER)

        if api_key:
            logger.info(
                "api_request",
                path=request.url.path,
                method=request.method,
                api_key_prefix=api_key[:10]
            )

        response = await call_next(request)
        return response
