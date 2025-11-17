"""Django Admin for Support Management with Django Unfold.

Rich admin interface with SLA tracking, auto-assignment, and bulk actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.db.models import Count
from django.http import HttpRequest
from django.utils import timezone
from django.utils.html import format_html
from django.utils.safestring import SafeString
from unfold.admin import ModelAdmin, TabularInline
from unfold.decorators import display

from .models import AccessRecoveryRequest, SupportTicket, TicketMessage
from .tasks import send_ticket_notification

if TYPE_CHECKING:
    from django.db.models import QuerySet


class TicketMessageInline(TabularInline):
    """Inline admin for ticket messages."""

    model = TicketMessage
    extra = 1
    fields: ClassVar = ["author", "message", "is_internal", "created_at"]
    readonly_fields: ClassVar = ["created_at"]


@admin.register(SupportTicket)
class SupportTicketAdmin(ModelAdmin):
    """Admin interface for support tickets with Unfold theme."""

    list_display: ClassVar = [
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

    list_filter: ClassVar = [
        "status",
        "priority",
        "category",
        "sla_breached",
        "created_at",
    ]

    search_fields: ClassVar = [
        "ticket_number",
        "subject",
        "description",
        "user__email",
        "user__first_name",
        "user__last_name",
    ]

    readonly_fields: ClassVar = [
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
            {"fields": ("subject", "description")},
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

    inlines: ClassVar = [TicketMessageInline]
    actions: ClassVar = ["assign_to_me", "mark_as_resolved", "escalate_priority", "send_reminder_email"]

    def get_queryset(self, request: HttpRequest) -> QuerySet[SupportTicket]:
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related("user", "assigned_to").annotate(message_count=Count("messages"))

    @display(description="Customer")
    def user_link(self, obj: SupportTicket) -> SafeString:
        """Display user with link to admin."""
        if obj.user:
            return format_html(
                '<a href="/admin/users/user/{}/">{}</a>',
                obj.user.id,
                obj.user.email,
            )
        return format_html("<span>No user</span>")

    @display(description="Category", label=True)
    def category_badge(self, obj: SupportTicket) -> SafeString:
        """Display category as badge."""
        badge_classes = {
            "technical": "info",
            "billing": "success",
            "access": "danger",
            "feature": "warning",
            "bug": "danger",
            "other": "secondary",
        }
        badge_class = badge_classes.get(obj.category, "secondary")
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            badge_class,
            obj.get_category_display(),
        )

    @display(description="Priority", label=True)
    def priority_badge(self, obj: SupportTicket) -> SafeString:
        """Display priority badge."""
        badge_classes = {
            "low": "secondary",
            "medium": "warning",
            "high": "danger",
            "critical": "danger",
        }
        badge_class = badge_classes.get(obj.priority, "secondary")
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            badge_class,
            obj.get_priority_display(),
        )

    @display(description="Status", label=True)
    def status_badge(self, obj: SupportTicket) -> SafeString:
        """Display status badge."""
        badge_classes = {
            "new": "info",
            "open": "warning",
            "pending": "warning",
            "in_progress": "info",
            "resolved": "success",
            "closed": "secondary",
            "reopened": "danger",
        }
        badge_class = badge_classes.get(obj.status, "secondary")
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            badge_class,
            obj.get_status_display(),
        )

    @display(description="SLA Status", label=True)
    def sla_indicator(self, obj: SupportTicket) -> SafeString:
        """Visual SLA status indicator."""
        if obj.status in ["resolved", "closed"]:
            if obj.sla_breached:
                return format_html('<span class="badge bg-danger">Breached</span>')
            return format_html('<span class="badge bg-success">Met</span>')

        time_left = obj.time_until_sla
        if time_left and time_left.total_seconds() < 0:
            return format_html('<span class="badge bg-danger">OVERDUE</span>')
        elif time_left and time_left.total_seconds() < 3600:
            return format_html(
                '<span class="badge bg-warning">{}m left</span>',
                int(time_left.total_seconds() / 60),
            )
        elif time_left:
            hours = int(time_left.total_seconds() / 3600)
            return format_html('<span class="badge bg-success">{}h left</span>', hours)
        return format_html("<span>-</span>")

    @admin.action(description="Assign selected tickets to me")
    def assign_to_me(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Assign tickets to current admin user."""
        count = 0
        for ticket in queryset:
            ticket.assign_to_agent(request.user)
            count += 1
        self.message_user(request, f"{count} tickets assigned to you.")

    @admin.action(description="Mark as resolved")
    def mark_as_resolved(self, request: HttpRequest, queryset: QuerySet[SupportTicket]) -> None:
        """Mark tickets as resolved."""
        count = 0
        for ticket in queryset:
            ticket.status = SupportTicket.Status.RESOLVED
            ticket.resolved_at = timezone.now()
            ticket.save()
            count += 1
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
            send_ticket_notification.delay(str(ticket.id), "reminder")
            count += 1
        self.message_user(request, f"Reminder emails queued for {count} tickets.")


@admin.register(TicketMessage)
class TicketMessageAdmin(ModelAdmin):
    """Admin interface for ticket messages."""

    list_display: ClassVar = [
        "ticket",
        "author",
        "message_preview",
        "is_internal",
        "is_system",
        "created_at",
    ]

    list_filter: ClassVar = [
        "is_internal",
        "is_system",
        "created_at",
    ]

    search_fields: ClassVar = [
        "message",
        "ticket__ticket_number",
        "author__email",
    ]

    readonly_fields: ClassVar = ["created_at"]

    @display(description="Message")
    def message_preview(self, obj: TicketMessage) -> str:
        """Display message preview."""
        return obj.message[:100] + ("..." if len(obj.message) > 100 else "")


@admin.register(AccessRecoveryRequest)
class AccessRecoveryRequestAdmin(ModelAdmin):
    """Admin interface for access recovery requests."""

    list_display: ClassVar = [
        "user",
        "request_type",
        "status_badge",
        "verification_method",
        "processed_by",
        "created_at",
    ]

    list_filter: ClassVar = [
        "status",
        "request_type",
        "verification_method",
        "created_at",
    ]

    search_fields: ClassVar = [
        "user__email",
        "admin_notes",
    ]

    readonly_fields: ClassVar = [
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

    actions: ClassVar = ["approve_requests", "reject_requests"]

    @display(description="Status", label=True)
    def status_badge(self, obj: AccessRecoveryRequest) -> SafeString:
        """Display status badge."""
        badge_classes = {
            "pending": "warning",
            "verified": "info",
            "approved": "success",
            "rejected": "danger",
            "completed": "secondary",
        }
        badge_class = badge_classes.get(obj.status, "secondary")
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            badge_class,
            obj.get_status_display(),
        )

    @admin.action(description="Approve selected requests")
    def approve_requests(self, request: HttpRequest, queryset) -> None:
        """Approve access recovery requests."""
        count = 0
        for recovery in queryset:
            recovery.approve(request.user, "Approved via admin bulk action")
            count += 1
        self.message_user(request, f"{count} requests approved.")

    @admin.action(description="Reject selected requests")
    def reject_requests(self, request: HttpRequest, queryset) -> None:
        """Reject access recovery requests."""
        count = 0
        for recovery in queryset:
            recovery.reject(request.user, "Rejected via admin bulk action")
            count += 1
        self.message_user(request, f"{count} requests rejected.")
