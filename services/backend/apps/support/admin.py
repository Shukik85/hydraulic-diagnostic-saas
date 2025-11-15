"""Django Admin for Support Management.

Rich admin interface with SLA tracking, auto-assignment, and bulk actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.contrib import admin
from django.db.models import Count, Q
from django.http import HttpRequest
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import AccessRecoveryRequest, SupportTicket, TicketMessage
from .tasks import send_ticket_notification

if TYPE_CHECKING:
    from django.db.models import QuerySet


class TicketMessageInline(admin.TabularInline[TicketMessage, SupportTicket]):
    """Inline admin for ticket messages."""

    model = TicketMessage
    extra = 1
    fields = ["author", "message", "is_internal", "created_at"]
    readonly_fields = ["created_at"]


@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin[SupportTicket]):
    """Admin interface for support tickets.

    Features:
    - SLA tracking with visual indicators
    - Auto-assignment to agents
    - Bulk actions (assign, close, escalate)
    - Inline message thread
    - Email notifications
    """

    list_display = [
        "ticket_number",
        "subject",
        "user_link",
        "category_badge",
        "priority_badge",
        "status_badge",
        "assigned_to",
        "sla_indicator",
        "created_at",
    ]

    list_filter = [
        "status",
        "priority",
        "category",
        "sla_breached",
        "created_at",
    ]

    search_fields = [
        "ticket_number",
        "subject",
        "description",
        "user__email",
        "user__first_name",
        "user__last_name",
    ]

    readonly_fields = [
        "ticket_number",
        "sla_due_date",
        "sla_breached",
        "resolved_at",
        "created_at",
        "updated_at",
    ]

    fieldsets = (
        (
            "Ticket Information",
            {
                "fields": (
                    "ticket_number",
                    "user",
                    "assigned_to",
                    "category",
                    "priority",
                    "status",
                )
            },
        ),
        (
            "Details",
            {
                "fields": ("subject", "description")
            },
        ),
        (
            "SLA Tracking",
            {
                "fields": ("sla_due_date", "sla_breached", "resolved_at"),
                "classes": ["collapse"],
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ["collapse"],
            },
        ),
    )

    inlines = [TicketMessageInline]

    actions = [
        "assign_to_me",
        "mark_as_resolved",
        "escalate_priority",
        "send_reminder_email",
    ]

    def get_queryset(self, request: HttpRequest) -> QuerySet[SupportTicket]:
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related("user", "assigned_to").annotate(
            message_count=Count("messages")
        )

    def user_link(self, obj: SupportTicket) -> SafeString:
        """Display user with link to admin."""
        return format_html(
            '<a href="/admin/users/user/{}/">{}</a>',
            obj.user.id,
            obj.user.email,
        )

    user_link.short_description = "Customer"  # type: ignore[attr-defined]

    def category_badge(self, obj: SupportTicket) -> SafeString:
        """Display category as colored badge."""
        colors = {
            "technical": "#3498db",
            "billing": "#2ecc71",
            "access": "#e74c3c",
            "feature": "#9b59b6",
            "bug": "#e67e22",
            "other": "#95a5a6",
        }
        color = colors.get(obj.category, "#95a5a6")
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_category_display(),
        )

    category_badge.short_description = "Category"  # type: ignore[attr-defined]

    def priority_badge(self, obj: SupportTicket) -> SafeString:
        """Display priority with urgency indicator."""
        colors = {
            "low": "#95a5a6",
            "medium": "#f39c12",
            "high": "#e67e22",
            "critical": "#c0392b",
        }
        color = colors.get(obj.priority, "#95a5a6")
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            obj.get_priority_display().split(" ")[0],  # Just the priority name
        )

    priority_badge.short_description = "Priority"  # type: ignore[attr-defined]

    def status_badge(self, obj: SupportTicket) -> SafeString:
        """Display status with color coding."""
        colors = {
            "new": "#3498db",
            "open": "#1abc9c",
            "pending": "#f39c12",
            "in_progress": "#9b59b6",
            "resolved": "#27ae60",
            "closed": "#7f8c8d",
            "reopened": "#e74c3c",
        }
        color = colors.get(obj.status, "#95a5a6")
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def sla_indicator(self, obj: SupportTicket) -> SafeString:
        """Visual SLA status indicator."""
        if obj.status in ["resolved", "closed"]:
            if obj.sla_breached:
                return format_html(
                    '<span style="color: #c0392b;">✗ Breached</span>'
                )
            return format_html(
                '<span style="color: #27ae60;">✓ Met</span>'
            )

        time_left = obj.time_until_sla
        if time_left and time_left.total_seconds() < 0:
            return format_html(
                '<span style="color: #c0392b; font-weight: bold;">⚠ OVERDUE</span>'
            )
        elif time_left and time_left.total_seconds() < 3600:  # < 1 hour
            return format_html(
                '<span style="color: #e67e22; font-weight: bold;">⚠ {}</span>',
                f"{int(time_left.total_seconds() / 60)}m left",
            )
        elif time_left:
            hours = int(time_left.total_seconds() / 3600)
            return format_html(
                '<span style="color: #27ae60;">✓ {}h left</span>',
                hours,
            )
        return format_html('<span>-</span>')

    sla_indicator.short_description = "SLA Status"  # type: ignore[attr-defined]

    # Admin actions
    @admin.action(description="Assign selected tickets to me")
    def assign_to_me(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Assign tickets to current admin user."""
        count = 0
        for ticket in queryset:
            ticket.assign_to_agent(request.user)  # type: ignore[arg-type]
            count += 1
        self.message_user(request, f"{count} tickets assigned to you.")

    @admin.action(description="Mark as resolved")
    def mark_as_resolved(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Mark tickets as resolved."""
        count = queryset.update(
            status=SupportTicket.Status.RESOLVED,
            resolved_at=timezone.now(),
        )
        self.message_user(request, f"{count} tickets marked as resolved.")

    @admin.action(description="Escalate to HIGH priority")
    def escalate_priority(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Escalate tickets to high priority."""
        count = queryset.update(priority=SupportTicket.Priority.HIGH)
        self.message_user(request, f"{count} tickets escalated to HIGH priority.")

    @admin.action(description="Send reminder email")
    def send_reminder_email(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Send reminder emails for selected tickets."""
        count = 0
        for ticket in queryset:
            send_ticket_notification.delay(ticket.id, "reminder")
            count += 1
        self.message_user(request, f"Reminder emails queued for {count} tickets.")


@admin.register(TicketMessage)
class TicketMessageAdmin(admin.ModelAdmin[TicketMessage]):
    """Admin interface for ticket messages."""

    list_display = [
        "ticket",
        "author",
        "message_preview",
        "is_internal",
        "is_system",
        "created_at",
    ]

    list_filter = [
        "is_internal",
        "is_system",
        "created_at",
    ]

    search_fields = [
        "message",
        "ticket__ticket_number",
        "author__email",
    ]

    readonly_fields = ["created_at"]

    def message_preview(self, obj: TicketMessage) -> str:
        """Display message preview."""
        return obj.message[:100] + ("..." if len(obj.message) > 100 else "")

    message_preview.short_description = "Message"  # type: ignore[attr-defined]


@admin.register(AccessRecoveryRequest)
class AccessRecoveryRequestAdmin(admin.ModelAdmin[AccessRecoveryRequest]):
    """Admin interface for access recovery requests."""

    list_display = [
        "user",
        "request_type",
        "status_badge",
        "verification_method",
        "processed_by",
        "created_at",
        "actions_column",
    ]

    list_filter = [
        "status",
        "request_type",
        "verification_method",
        "created_at",
    ]

    search_fields = [
        "user__email",
        "admin_notes",
    ]

    readonly_fields = [
        "created_at",
        "processed_at",
        "processed_by",
    ]

    fieldsets = (
        (
            "Request Information",
            {
                "fields": (
                    "user",
                    "request_type",
                    "status",
                    "verification_method",
                )
            },
        ),
        (
            "Processing",
            {
                "fields": (
                    "admin_notes",
                    "processed_by",
                    "processed_at",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at",),
                "classes": ["collapse"],
            },
        ),
    )

    actions = ["approve_requests", "reject_requests"]

    def status_badge(self, obj: AccessRecoveryRequest) -> SafeString:
        """Display status badge."""
        colors = {
            "pending": "#3498db",
            "verified": "#1abc9c",
            "approved": "#27ae60",
            "rejected": "#c0392b",
            "completed": "#7f8c8d",
        }
        color = colors.get(obj.status, "#95a5a6")
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def actions_column(self, obj: AccessRecoveryRequest) -> SafeString:
        """Display action buttons."""
        if obj.status == "pending":
            return format_html(
                '<a class="button" style="margin-right: 5px; background: #27ae60; color: white;" '
                'href="#" onclick="return confirm(\'Approve this request?\')">Approve</a>'
                '<a class="button" style="background: #c0392b; color: white;" '
                'href="#" onclick="return confirm(\'Reject this request?\')">Reject</a>'
            )
        return format_html('<span>-</span>')

    actions_column.short_description = "Actions"  # type: ignore[attr-defined]

    @admin.action(description="Approve selected requests")
    def approve_requests(
        self, request: HttpRequest, queryset: QuerySet[AccessRecoveryRequest]
    ) -> None:
        """Approve access recovery requests."""
        count = 0
        for recovery in queryset:
            recovery.approve(request.user, "Approved via admin bulk action")  # type: ignore[arg-type]
            count += 1
        self.message_user(request, f"{count} requests approved.")

    @admin.action(description="Reject selected requests")
    def reject_requests(
        self, request: HttpRequest, queryset: QuerySet[AccessRecoveryRequest]
    ) -> None:
        """Reject access recovery requests."""
        count = 0
        for recovery in queryset:
            recovery.reject(request.user, "Rejected via admin bulk action")  # type: ignore[arg-type]
            count += 1
        self.message_user(request, f"{count} requests rejected.")


# Django timezone import
from django.utils import timezone
