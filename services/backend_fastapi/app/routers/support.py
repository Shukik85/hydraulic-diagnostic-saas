"""
Customer support endpoints
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

from app.dependencies import get_current_user
from app.models.support import SupportTicket
from app.models.user import User
from app.tasks.email import send_support_ticket_notification
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/support", tags=["Support"])


class SupportTicketCreate(BaseModel):
    subject: str = Field(..., min_length=5, max_length=255)
    message: str = Field(..., min_length=10, max_length=5000)
    priority: Literal["low", "medium", "high"] = "medium"
    category: str | None = Field(None, description="ticket_category")


class SupportTicketResponse(BaseModel):
    id: uuid.UUID
    subject: str
    message: str
    priority: str
    status: str
    created_at: datetime
    updated_at: datetime | None


class SupportTicketUpdate(BaseModel):
    message: str | None = None


# ============= Endpoints =============


@router.post("/tickets", response_model=SupportTicketResponse, status_code=201)
async def create_support_ticket(
    ticket_data: SupportTicketCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),  # noqa: B008
):
    """
    Create a new support ticket

    Automatically notifies support team
    """
    # Create ticket
    ticket = await SupportTicket.create(
        user_id=current_user.id,
        subject=ticket_data.subject,
        message=ticket_data.message,
        priority=ticket_data.priority,
        category=ticket_data.category,
        status="open",
    )

    # Notify support team (async)
    background_tasks.add_task(
        send_support_ticket_notification,
        ticket_id=ticket.id,
        user_email=current_user.email,
        subject=ticket_data.subject,
        priority=ticket_data.priority,
    )

    return ticket


@router.get("/tickets", response_model=list[SupportTicketResponse])
async def get_my_tickets(
    status: str | None = None,
    current_user: User = Depends(get_current_user),  # noqa: B008
):
    """
    Get all support tickets for current user
    """
    query = {"user_id": current_user.id}
    if status:
        query["status"] = status

    return await SupportTicket.find(query)


@router.get("/tickets/{ticket_id}", response_model=SupportTicketResponse)
async def get_ticket(
    ticket_id: uuid.UUID,
    current_user: User = Depends(get_current_user),  # noqa: B008
):
    """
    Get specific support ticket
    """
    ticket = await SupportTicket.get(ticket_id)

    if not ticket:
        raise HTTPException(404, "Ticket not found")

    if ticket.user_id != current_user.id:
        raise HTTPException(403, "Access denied")

    return ticket


@router.patch("/tickets/{ticket_id}", response_model=SupportTicketResponse)
async def update_ticket(
    ticket_id: uuid.UUID,
    update_data: SupportTicketUpdate,
    current_user: User = Depends(get_current_user),  # noqa: B008
):
    """
    Add message/update to existing ticket
    """
    ticket = await SupportTicket.get(ticket_id)

    if not ticket:
        raise HTTPException(404, "Ticket not found")

    if ticket.user_id != current_user.id:
        raise HTTPException(403, "Access denied")

    if update_data.message:
        # Append message to ticket history
        ticket.message += (
            f"\n\n--- User update ({datetime.now(UTC)}) ---\n{update_data.message}"
        )
        ticket.updated_at = datetime.now(UTC)
        await ticket.save()

    return ticket
