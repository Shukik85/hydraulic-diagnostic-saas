"""Django Admin for Support Management with Friendly UX.

Rich admin interface with SLA tracking, auto-assignment, and bulk actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.db.models import Count
from django.http import HttpRequest
from django.templatetags.static import static
from django.utils import timezone
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import AccessRecoveryRequest, SupportTicket, TicketMessage
from .tasks import send_ticket_notification

if TYPE_CHECKING:
    from django.db.models import QuerySet


class TicketMessageInline(admin.TabularInline):
    """Inline admin for ticket messages."""

    model = TicketMessage
    extra = 1
    fields: ClassVar = ["author", "message", "is_internal", "created_at"]
    readonly_fields: ClassVar = ["created_at"]


@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin):
    """Admin interface for support tickets with Friendly UX.

    Features:
    - SLA tracking with visual indicators
    - Auto-assignment to agents
    - Bulk actions (assign, close, escalate)
    - Inline message thread
    - Email notifications
    - FriendlyUX badges with SVG icons
    """

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

    actions: ClassVar = [
        "assign_to_me",
        "mark_as_resolved",
        "escalate_priority",
        "send_reminder_email",
    ]

    def get_queryset(self, request: HttpRequest) -> QuerySet[SupportTicket]:
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related("user", "assigned_to").annotate(message_count=Count("messages"))

    def user_link(self, obj: SupportTicket) -> SafeString:
        """Display user with link to admin."""
        if obj.user:
            return format_html(
                '<a href="/admin/users/user/{}/">{}</a>',
                obj.user.id,
                obj.user.email,
            )
        return format_html("<span>No user</span>")

    user_link.short_description = "Customer"

    def category_badge(self, obj: SupportTicket) -> SafeString:
        """Display category as FriendlyUX badge."""
        badge_classes = {
            "technical": "BadgeInfo",
            "billing": "BadgeSuccess",
            "access": "BadgeError",
            "feature": "BadgeWarning",
            "bug": "BadgeError",
            "other": "BadgeMuted",
        }
        badge_class = badge_classes.get(obj.category, "BadgeMuted")
        return format_html(
            '<span class="Badge {}">{}</span>',
            badge_class,
            obj.get_category_display(),
        )

    category_badge.short_description = "Category"

    def priority_badge(self, obj: SupportTicket) -> SafeString:
        """Display priority with FriendlyUX badge and icon."""
        badge_classes = {
            "low": "BadgeMuted",
            "medium": "BadgeWarning",
            "high": "BadgeError",
            "critical": "BadgeError",
        }
        badge_class = badge_classes.get(obj.priority, "BadgeMuted")

        icon_name = {
            "low": "icon-arrow-down",
            "medium": "icon-minus",
            "high": "icon-arrow-up",
            "critical": "icon-alert",
        }.get(obj.priority, "icon-minus")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_priority_display().split(" ")[0],
        )

    priority_badge.short_description = "Priority"

    def status_badge(self, obj: SupportTicket) -> SafeString:
        """Display status with FriendlyUX badge and icon."""
        badge_classes = {
            "new": "BadgeInfo",
            "open": "BadgeWarning",
            "pending": "BadgeWarning",
            "in_progress": "BadgeInfo",
            "resolved": "BadgeSuccess",
            "closed": "BadgeMuted",
            "reopened": "BadgeError",
        }
        badge_class = badge_classes.get(obj.status, "BadgeMuted")

        icon_name = {
            "new": "icon-star",
            "open": "icon-circle",
            "pending": "icon-clock",
            "in_progress": "icon-refresh",
            "resolved": "icon-check",
            "closed": "icon-x",
            "reopened": "icon-alert",
        }.get(obj.status, "icon-circle")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"

    def sla_indicator(self, obj: SupportTicket) -> SafeString:
        """Visual SLA status indicator with FriendlyUX badges."""
        if obj.status in ["resolved", "closed"]:
            if obj.sla_breached:
                return format_html(
                    '<span class="Badge BadgeError">'
                    '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                    '<use href="{}#icon-x"></use></svg> Breached'
                    "</span>",
                    static("admin/icons/icons-sprite.svg"),
                )
            return format_html(
                '<span class="Badge BadgeSuccess">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-check"></use></svg> Met'
                "</span>",
                static("admin/icons/icons-sprite.svg"),
            )

        time_left = obj.time_until_sla
        if time_left and time_left.total_seconds() < 0:
            return format_html(
                '<span class="Badge BadgeError">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-alert"></use></svg> OVERDUE'
                "</span>",
                static("admin/icons/icons-sprite.svg"),
            )
        elif time_left and time_left.total_seconds() < 3600:
            return format_html(
                '<span class="Badge BadgeWarning">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-clock"></use></svg> {}'
                "</span>",
                static("admin/icons/icons-sprite.svg"),
                f"{int(time_left.total_seconds() / 60)}m left",
            )
        elif time_left:
            hours = int(time_left.total_seconds() / 3600)
            return format_html(
                '<span class="Badge BadgeSuccess">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-check"></use></svg> {}h left'
                "</span>",
                static("admin/icons/icons-sprite.svg"),
                hours,
            )
        return format_html("<span>-</span>")

    sla_indicator.short_description = "SLA Status"

    # Admin actions
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
class TicketMessageAdmin(admin.ModelAdmin):
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

    def message_preview(self, obj: TicketMessage) -> str:
        """Display message preview."""
        return obj.message[:100] + ("..." if len(obj.message) > 100 else "")

    message_preview.short_description = "Message"


@admin.register(AccessRecoveryRequest)
class AccessRecoveryRequestAdmin(admin.ModelAdmin):
    """Admin interface for access recovery requests."""

    list_display: ClassVar = [
        "user",
        "request_type",
        "status_badge",
        "verification_method",
        "processed_by",
        "created_at",
        "actions_column",
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

    def status_badge(self, obj: AccessRecoveryRequest) -> SafeString:
        """Display status badge with FriendlyUX."""
        badge_classes = {
            "pending": "BadgeWarning",
            "verified": "BadgeInfo",
            "approved": "BadgeSuccess",
            "rejected": "BadgeError",
            "completed": "BadgeMuted",
        }
        badge_class = badge_classes.get(obj.status, "BadgeMuted")
        return format_html(
            '<span class="Badge {}">{}</span>',
            badge_class,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"

    def actions_column(self, obj: AccessRecoveryRequest) -> SafeString:
        """Display action buttons with FriendlyUX."""
        if obj.status == "pending":
            return format_html(
                '<a class="Btn" style="padding: 6px 12px; font-size: 12px; margin-right: 4px;" '
                'href="#" onclick="return confirm(\'Approve this request?\')">' 
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-check"></use></svg> Approve'
                "</a>"
                '<a class="Btn" style="padding: 6px 12px; font-size: 12px; background: var(--color-error);" '
                'href="#" onclick="return confirm(\'Reject this request?\')">' 
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none;">'
                '<use href="{}#icon-x"></use></svg> Reject'
                "</a>",
                static("admin/icons/icons-sprite.svg"),
                static("admin/icons/icons-sprite.svg"),
            )
        return format_html("<span>-</span>")

    actions_column.short_description = "Actions"

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
