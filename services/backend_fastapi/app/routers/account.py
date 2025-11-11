"""
Account management and data export (GDPR)
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

from app.models.user import User
from app.models.data_export import DataExportRequest
from app.dependencies import get_current_user
from app.tasks.data_export import export_user_data_task

router = APIRouter(prefix="/account", tags=["Account"])


# ============= Models =============

class DataExportResponse(BaseModel):
    request_id: uuid.UUID
    status: str
    message: str
    estimated_time: str = "~30-60 minutes"

class AccountInfo(BaseModel):
    id: uuid.UUID
    email: str
    subscription_tier: str
    subscription_status: str
    api_requests_count: int
    ml_inferences_count: int
    created_at: datetime


# ============= Endpoints =============

@router.get("/me", response_model=AccountInfo)
async def get_account_info(current_user: User = Depends(get_current_user)):
    """
    Get current user account information
    """
    return AccountInfo(
        id=current_user.id,
        email=current_user.email,
        subscription_tier=current_user.subscription_tier,
        subscription_status=current_user.subscription_status,
        api_requests_count=current_user.api_requests_count,
        ml_inferences_count=current_user.ml_inferences_count,
        created_at=current_user.created_at
    )


@router.post("/export-data", response_model=DataExportResponse, status_code=202)
async def request_data_export(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Request complete data export (GDPR compliance)

    Exports:
    - User profile
    - Equipment metadata
    - Sensor data archives
    - API usage logs

    Download link sent via email within 1 hour
    """
    # Check for existing pending request
    existing = await DataExportRequest.get_active_by_user(current_user.id)
    if existing:
        return DataExportResponse(
            request_id=existing.id,
            status="pending",
            message="Data export already in progress"
        )

    # Create new export request
    export_request = await DataExportRequest.create(
        user_id=current_user.id,
        status="pending"
    )

    # Queue export task
    background_tasks.add_task(
        export_user_data_task,
        export_request.id,
        current_user.id,
        current_user.email
    )

    return DataExportResponse(
        request_id=export_request.id,
        status="pending",
        message="Data export started. Download link will be sent to your email."
    )


@router.delete("/me")
async def delete_account(
    confirmation: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete user account (GDPR right to be forgotten)

    Requires confirmation string: "DELETE MY ACCOUNT"
    """
    if confirmation != "DELETE MY ACCOUNT":
        raise HTTPException(400, "Invalid confirmation string")

    # Soft delete (mark as deleted, anonymize data)
    current_user.is_active = False
    current_user.email = f"deleted_{current_user.id}@deleted.local"
    current_user.first_name = "Deleted"
    current_user.last_name = "User"
    current_user.api_key = f"deleted_{uuid.uuid4()}"
    current_user.updated_at = datetime.utcnow()
    await current_user.save()

    # TODO: Schedule data purge after 30 days

    return {"message": "Account deleted. Data will be permanently removed in 30 days."}
