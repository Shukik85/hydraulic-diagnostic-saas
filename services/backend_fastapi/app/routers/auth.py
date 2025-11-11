"""
Authentication and password reset endpoints
"""

import secrets
from datetime import UTC, datetime

from app.core.redis import redis_client
from app.core.security import hash_password
from app.dependencies import get_current_user
from app.models.user import User
from app.tasks.email import send_new_api_key_email, send_password_reset_email
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["Authentication"])


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Body(..., min_length=8)


class APIKeyResetResponse(BaseModel):
    message: str
    api_key_preview: str


# ============= Endpoints =============


@router.post("/password-reset-request", status_code=202)
async def request_password_reset(
    data: PasswordResetRequest, background_tasks: BackgroundTasks
):
    """
    Request password reset link via email

    Returns 202 even if email doesn't exist (security best practice)
    """
    user = await User.get_by_email(data.email)

    if not user:
        # Don't reveal that email doesn't exist
        return {"message": "If the email exists, a reset link has been sent"}

    # Generate reset token
    reset_token = secrets.token_urlsafe(32)

    # Store token in Redis (expires in 1 hour)
    await redis_client.setex(
        f"password_reset:{reset_token}",
        3600,  # 1 hour TTL
        str(user.id),
    )

    # Send email asynchronously
    reset_link = f"https://yourdomain.com/reset-password?token={reset_token}"
    background_tasks.add_task(
        send_password_reset_email, user.email, user.first_name or "User", reset_link
    )

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset-confirm")
async def confirm_password_reset(data: PasswordResetConfirm):
    """
    Confirm password reset with token
    """
    # Validate token
    user_id = await redis_client.get(f"password_reset:{data.token}")

    if not user_id:
        raise HTTPException(400, "Invalid or expired reset token")

    # Get user
    user = await User.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    # Update password
    user.password = hash_password(data.new_password)
    user.updated_at = datetime.now(UTC)
    await user.save()

    # Delete token
    await redis_client.delete(f"password_reset:{data.token}")

    return {"message": "Password reset successful"}


@router.post("/api-key-reset", response_model=APIKeyResetResponse)
async def reset_api_key(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),  # noqa: B008
):
    """
    Regenerate API key (invalidates old key)

    Requires authentication (Bearer token)
    """
    # Generate new API key
    new_api_key = f"hyd_{secrets.token_urlsafe(32)}"

    # Update user
    current_user.api_key = new_api_key
    current_user.updated_at = datetime.now(UTC)
    await current_user.save()

    # Send new key via email
    background_tasks.add_task(
        send_new_api_key_email,
        current_user.email,
        current_user.first_name or "User",
        new_api_key,
    )

    return APIKeyResetResponse(
        message="New API key generated and sent to your email",
        api_key_preview=f"{new_api_key[:15]}...",
    )


@router.post("/verify-token")
async def verify_reset_token(token: str):
    """
    Verify if reset token is valid (for frontend validation)
    """
    user_id = await redis_client.get(f"password_reset:{token}")

    if not user_id:
        raise HTTPException(400, "Invalid or expired token")

    return {"valid": True}
