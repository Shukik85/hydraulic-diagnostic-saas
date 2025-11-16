"""Support management models with Python 3.14 type hints.

Fully typed models for ticket management, SLA tracking, and access recovery.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from django.conf import settings
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from apps.users.models import User


class SupportTicket(models.Model):
    """Support ticket for customer issues and requests.

    Tracks tickets with categories, priorities, SLA management, and status.

    Attributes:
        id: UUID primary key
        ticket_number: Human-readable ticket number (auto-generated)
        user: Customer who created the ticket
        assigned_to: Support agent assigned to this ticket
        category: Type of issue/request
        priority: Urgency level (affects SLA)
        status: Current ticket status
        subject: Brief summary of the issue
        description: Detailed description
        sla_due_date: When ticket should be resolved (based on priority)
        sla_breached: Whether SLA deadline was missed
        resolved_at: When ticket was resolved
        created_at: Ticket creation timestamp
        updated_at: Last update timestamp
    """

    # Category choices
    class Category(models.TextChoices):
        """Ticket category types."""

        TECHNICAL = "technical", "Technical Issue"
        BILLING = "billing", "Billing Question"
        ACCESS = "access", "Account Access"
        FEATURE = "feature", "Feature Request"
        BUG = "bug", "Bug Report"
        OTHER = "other", "Other"

    # Priority choices (affects SLA)
    class Priority(models.TextChoices):
        """Ticket priority levels."""

        LOW = "low", "Low (72h SLA)"
        MEDIUM = "medium", "Medium (24h SLA)"
        HIGH = "high", "High (8h SLA)"
        CRITICAL = "critical", "Critical (2h SLA)"

    # Status choices
    class Status(models.TextChoices):
        """Ticket status workflow."""

        NEW = "new", "New"
        OPEN = "open", "Open"
        PENDING = "pending", "Pending Customer Response"
        IN_PROGRESS = "in_progress", "In Progress"
        RESOLVED = "resolved", "Resolved"
        CLOSED = "closed", "Closed"
        REOPENED = "reopened", "Reopened"

    # SLA timeframes (hours)
    SLA_TIMEFRAMES: ClassVar[dict[str, int]] = {
        Priority.LOW: 72,
        Priority.MEDIUM: 24,
        Priority.HIGH: 8,
        Priority.CRITICAL: 2,
    }

    id: uuid.UUID = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    ticket_number: str = models.CharField(
        max_length=20,
        unique=True,
        editable=False,
        help_text="Human-readable ticket number (e.g., TKT-2024-00123)",
    )

    user: User = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="support_tickets",
        help_text="Customer who created this ticket",
    )

    assigned_to: User | None = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="assigned_tickets",
        null=True,
        blank=True,
        limit_choices_to={"is_staff": True},
        help_text="Support agent assigned to this ticket",
    )

    category: str = models.CharField(
        max_length=20,
        choices=Category.choices,
        default=Category.OTHER,
        help_text="Type of issue or request",
    )

    priority: str = models.CharField(
        max_length=20,
        choices=Priority.choices,
        default=Priority.MEDIUM,
        help_text="Urgency level (affects SLA)",
    )

    status: str = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.NEW,
        db_index=True,
        help_text="Current ticket status",
    )

    subject: str = models.CharField(
        max_length=255,
        help_text="Brief summary of the issue",
    )

    description: str = models.TextField(
        help_text="Detailed description of the issue",
    )

    sla_due_date: datetime = models.DateTimeField(
        help_text="When ticket should be resolved (based on priority)",
    )

    sla_breached: bool = models.BooleanField(
        default=False,
        help_text="Whether SLA deadline was missed",
    )

    resolved_at: datetime | None = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When ticket was resolved",
    )

    created_at: datetime = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    updated_at: datetime = models.DateTimeField(
        auto_now=True,
    )

    # Type hints for Django internals
    if TYPE_CHECKING:
        DoesNotExist: ClassVar[type[Exception]]
        MultipleObjectsReturned: ClassVar[type[Exception]]
        objects: ClassVar[models.Manager[SupportTicket]]
        messages: models.Manager[TicketMessage]

    class Meta:
        db_table = "support_tickets"
        verbose_name = "Support Ticket"
        verbose_name_plural = "Support Tickets"
        ordering: ClassVar[list[str]] = ["-created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["-created_at", "status"], name="ticket_time_status_idx"),
            models.Index(fields=["priority", "status"], name="ticket_pri_status_idx"),
            models.Index(fields=["sla_due_date"], name="ticket_sla_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.ticket_number}: {self.subject}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Save with auto-generated ticket number and SLA calculation."""
        # Generate ticket number if new
        if not self.ticket_number:
            self.ticket_number = self._generate_ticket_number()

        # Calculate SLA due date if new
        if not self.pk and not self.sla_due_date:
            self.sla_due_date = self._calculate_sla_due_date()

        # Check SLA breach
        self._check_sla_breach()

        super().save(*args, **kwargs)

    @staticmethod
    def _generate_ticket_number() -> str:
        """Generate unique ticket number (e.g., TKT-2024-00123)."""
        from datetime import datetime

        year = datetime.now().year
        # Get last ticket number for current year
        last_ticket = (
            SupportTicket.objects.filter(ticket_number__startswith=f"TKT-{year}")
            .order_by("-ticket_number")
            .first()
        )

        if last_ticket:
            # Extract sequence number and increment
            seq = int(last_ticket.ticket_number.split("-")[-1]) + 1
        else:
            seq = 1

        return f"TKT-{year}-{seq:05d}"

    def _calculate_sla_due_date(self) -> datetime:
        """Calculate SLA due date based on priority."""
        hours = self.SLA_TIMEFRAMES[self.priority]
        return timezone.now() + timedelta(hours=hours)

    def _check_sla_breach(self) -> None:
        """Check if SLA has been breached."""
        if self.status not in [self.Status.RESOLVED, self.Status.CLOSED]:
            if timezone.now() > self.sla_due_date:
                self.sla_breached = True

    @property
    def time_until_sla(self) -> timedelta | None:
        """Time remaining until SLA breach.

        Returns:
            Timedelta if ticket is open, None if resolved/closed
        """
        if self.status in [self.Status.RESOLVED, self.Status.CLOSED]:
            return None
        return self.sla_due_date - timezone.now()

    @property
    def is_overdue(self) -> bool:
        """Check if ticket is overdue."""
        return self.sla_breached

    def assign_to_agent(self, agent: User) -> None:
        """Assign ticket to support agent.

        Args:
            agent: User with is_staff=True
        """
        if not agent.is_staff:
            raise ValueError("Can only assign to staff members")

        self.assigned_to = agent
        if self.status == self.Status.NEW:
            self.status = self.Status.OPEN
        self.save()

    def resolve(self) -> None:
        """Mark ticket as resolved."""
        self.status = self.Status.RESOLVED
        self.resolved_at = timezone.now()
        self.save()


class TicketMessage(models.Model):
    """Message in a support ticket thread.

    Represents a single message in the conversation between customer and support.

    Attributes:
        id: UUID primary key
        ticket: Parent support ticket
        author: User who wrote this message (customer or agent)
        message: Message content
        is_internal: Whether message is internal note (not visible to customer)
        is_system: Whether message is auto-generated system message
        created_at: Message timestamp
    """

    id: uuid.UUID = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    ticket: SupportTicket = models.ForeignKey(
        SupportTicket,
        on_delete=models.CASCADE,
        related_name="messages",
        help_text="Parent support ticket",
    )

    author: User = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ticket_messages",
        help_text="Message author (customer or agent)",
    )

    message: str = models.TextField(
        help_text="Message content",
    )

    is_internal: bool = models.BooleanField(
        default=False,
        help_text="Internal note (not visible to customer)",
    )

    is_system: bool = models.BooleanField(
        default=False,
        help_text="Auto-generated system message",
    )

    created_at: datetime = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    # Type hints
    if TYPE_CHECKING:
        DoesNotExist: ClassVar[type[Exception]]
        MultipleObjectsReturned: ClassVar[type[Exception]]
        objects: ClassVar[models.Manager[TicketMessage]]

    class Meta:
        db_table = "ticket_messages"
        verbose_name = "Ticket Message"
        verbose_name_plural = "Ticket Messages"
        ordering: ClassVar[list[str]] = ["created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["ticket", "created_at"], name="msg_ticket_time_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.author.email}: {self.message[:50]}"


class AccessRecoveryRequest(models.Model):
    """Request to recover account access.

    Handles password resets and account recovery workflows.

    Attributes:
        id: UUID primary key
        user: User requesting access recovery
        request_type: Type of recovery (password, 2FA, etc.)
        status: Current request status
        verification_method: How user was verified
        admin_notes: Internal notes for review
        processed_by: Admin who processed the request
        processed_at: When request was processed
        created_at: Request creation timestamp
    """

    # Request type choices
    class RequestType(models.TextChoices):
        """Access recovery request types."""

        PASSWORD_RESET = "password_reset", "Password Reset"
        TWO_FA_RESET = "2fa_reset", "2FA Reset"
        ACCOUNT_UNLOCK = "account_unlock", "Account Unlock"
        EMAIL_CHANGE = "email_change", "Email Change"

    # Status choices
    class Status(models.TextChoices):
        """Request status workflow."""

        PENDING = "pending", "Pending Review"
        VERIFIED = "verified", "Identity Verified"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        COMPLETED = "completed", "Completed"

    # Verification method choices
    class VerificationMethod(models.TextChoices):
        """How user identity was verified."""

        EMAIL = "email", "Email Verification"
        SMS = "sms", "SMS Verification"
        ID_DOCUMENT = "id_document", "ID Document"
        SUPPORT_CALL = "support_call", "Support Call"
        BILLING_INFO = "billing_info", "Billing Information"

    id: uuid.UUID = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    user: User = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="access_recovery_requests",
        help_text="User requesting access recovery",
    )

    request_type: str = models.CharField(
        max_length=20,
        choices=RequestType.choices,
        help_text="Type of access recovery",
    )

    status: str = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
        help_text="Current request status",
    )

    verification_method: str | None = models.CharField(
        max_length=20,
        choices=VerificationMethod.choices,
        null=True,
        blank=True,
        help_text="How user identity was verified",
    )

    admin_notes: str = models.TextField(
        blank=True,
        default="",
        help_text="Internal notes for review",
    )

    processed_by: User | None = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="processed_recovery_requests",
        null=True,
        blank=True,
        limit_choices_to={"is_staff": True},
        help_text="Admin who processed the request",
    )

    processed_at: datetime | None = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When request was processed",
    )

    created_at: datetime = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    # Type hints
    if TYPE_CHECKING:
        DoesNotExist: ClassVar[type[Exception]]
        MultipleObjectsReturned: ClassVar[type[Exception]]
        objects: ClassVar[models.Manager[AccessRecoveryRequest]]

    class Meta:
        db_table = "access_recovery_requests"
        verbose_name = "Access Recovery Request"
        verbose_name_plural = "Access Recovery Requests"
        ordering: ClassVar[list[str]] = ["-created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["-created_at", "status"], name="recovery_time_status_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.user.email} - {self.get_request_type_display()}"

    def approve(self, admin: User, notes: str = "") -> None:
        """Approve access recovery request.

        Args:
            admin: Admin approving the request
            notes: Additional notes
        """
        self.status = self.Status.APPROVED
        self.processed_by = admin
        self.processed_at = timezone.now()
        if notes:
            self.admin_notes += f"\n[{timezone.now()}] Approved: {notes}"
        self.save()

    def reject(self, admin: User, reason: str) -> None:
        """Reject access recovery request.

        Args:
            admin: Admin rejecting the request
            reason: Reason for rejection
        """
        self.status = self.Status.REJECTED
        self.processed_by = admin
        self.processed_at = timezone.now()
        self.admin_notes += f"\n[{timezone.now()}] Rejected: {reason}"
        self.save()
