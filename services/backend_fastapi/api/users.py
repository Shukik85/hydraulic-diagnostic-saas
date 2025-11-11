"""
User management API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from db.session import get_db
from models.user import User
from schemas.user import UserCreate, UserResponse
from services.auth_service import AuthService
from middleware.auth import get_current_user

router = APIRouter()
logger = structlog.get_logger()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register new user"""
    auth_service = AuthService(db)

    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user = await auth_service.create_user(
        email=user_data.email,
        password=user_data.password
    )

    logger.info("user_registered", user_id=str(user.id), email=user.email)
    return user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile"""
    return current_user


@router.get("/{user_id}/usage")
async def get_user_usage(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's API usage statistics"""
    # TODO: Implement usage tracking from Redis/TimescaleDB
    return {
        "user_id": user_id,
        "api_requests_today": 0,
        "gnn_inferences_this_month": 0,
        "quota_remaining": 100
    }
