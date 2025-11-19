"""Celery tasks for support management.

Asynchronous tasks for email notifications, SLA monitoring, and auto-assignment.
"""

from __future__ import annotations

from typing import Any

from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail

# Import models for tasks
from django.db import models
from django.template.loader import render_to_string
from django.utils import timezone

from .models import SupportTicket


@shared_task(bind=True, max_retries=3)
def send_ticket_notification(
    self: Any,
    ticket_id: str,
    notification_type: str,
) -> dict[str, Any]:
    """Send email notification for ticket events.

    Args:
        ticket_id: UUID of the ticket
        notification_type: Type of notification (created, updated, resolved, reminder)

    Returns:
        Dict with status and details
    """
    try:
        ticket = SupportTicket.objects.select_related("user", "assigned_to").get(id=ticket_id)

        # Determine recipients
        recipients = [ticket.user.email]
        if ticket.assigned_to:
            recipients.append(ticket.assigned_to.email)

        # Build email context
        context = {
            "ticket": ticket,
            "notification_type": notification_type,
            "support_url": f"{settings.FRONTEND_URL}/support/{ticket.ticket_number}",
        }

        # Select template based on notification type
        subject_map = {
            "created": f"Ticket Created: {ticket.ticket_number}",
            "updated": f"Ticket Updated: {ticket.ticket_number}",
            "resolved": f"Ticket Resolved: {ticket.ticket_number}",
            "reminder": f"Reminder: {ticket.ticket_number}",
        }

        subject = subject_map.get(notification_type, f"Ticket {ticket.ticket_number}")

        # Render email content
        html_message = render_to_string(
            f"support/email/{notification_type}.html",
            context,
        )

        # Send email
        send_mail(
            subject=subject,
            message="",  # Plain text version (optional)
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=recipients,
            html_message=html_message,
            fail_silently=False,
        )

        return {
            "status": "success",
            "ticket_number": ticket.ticket_number,
            "notification_type": notification_type,
            "recipients": recipients,
        }

    except SupportTicket.DoesNotExist:
        return {
            "status": "error",
            "message": f"Ticket {ticket_id} not found",
        }

    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))


@shared_task
def check_sla_breaches() -> dict[str, Any]:
    """Check for SLA breaches and send alerts.

    Runs periodically (e.g., every 30 minutes) to identify tickets
    approaching or exceeding SLA deadlines.

    Returns:
        Dict with breach statistics
    """
    from datetime import timedelta

    now = timezone.now()
    warning_threshold = now + timedelta(hours=1)  # Alert 1 hour before SLA

    # Find tickets approaching SLA
    approaching = SupportTicket.objects.filter(
        sla_due_date__lte=warning_threshold,
        sla_due_date__gte=now,
        status__in=["new", "open", "in_progress"],
        sla_breached=False,
    ).select_related("user", "assigned_to")

    # Find tickets that breached SLA
    breached = SupportTicket.objects.filter(
        sla_due_date__lt=now,
        status__in=["new", "open", "in_progress"],
        sla_breached=False,  # Not yet marked
    )

    # Mark breached tickets
    breached_count = breached.update(sla_breached=True)

    # Send alerts for approaching SLA
    for ticket in approaching:
        send_ticket_notification.delay(str(ticket.id), "sla_warning")

    # Send alerts for breached SLA
    for ticket in breached:
        send_ticket_notification.delay(str(ticket.id), "sla_breach")

    return {
        "approaching_sla": approaching.count(),
        "breached_sla": breached_count,
        "timestamp": now.isoformat(),
    }


@shared_task
def auto_assign_tickets() -> dict[str, int]:
    """Auto-assign new tickets to available agents.

    Uses round-robin distribution based on current workload.

    Returns:
        Dict with assignment statistics
    """
    from django.contrib.auth import get_user_model
    from django.db.models import Count

    user_model = get_user_model()

    # Get unassigned tickets
    unassigned = SupportTicket.objects.filter(
        assigned_to=None,
        status="new",
    ).order_by("created_at")

    # Get available agents (staff with least active tickets)
    agents = (
        user_model.objects.filter(
            is_staff=True,
            is_active=True,
        )
        .annotate(
            active_tickets=Count(
                "assigned_tickets",
                filter=models.Q(assigned_tickets__status__in=["new", "open", "in_progress"]),
            )
        )
        .order_by("active_tickets")
    )

    if not agents.exists():
        return {"assigned": 0, "skipped": unassigned.count()}

    assigned_count = 0

    for agent_index, ticket in enumerate(unassigned):
        agent = agents[agent_index % len(agents)]
        ticket.assign_to_agent(agent)

        # Send notification
        send_ticket_notification.delay(str(ticket.id), "assigned")

        assigned_count += 1

    return {
        "assigned": assigned_count,
        "total_agents": agents.count(),
    }
