"""Notifications admin with type-safe attributes."""

from __future__ import annotations

from typing import ClassVar

from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import EmailCampaign, Notification


@admin.register(EmailCampaign)
class EmailCampaignAdmin(admin.ModelAdmin):
    """Admin interface for email campaigns."""

    list_display: ClassVar[list[str]] = [
        "name",
        "subject",
        "status_badge",
        "recipients_count",
        "opened_count",
        "scheduled_for",
        "sent_at",
    ]
    list_filter: ClassVar[list[str]] = ["status", "target_tier"]
    search_fields: ClassVar[list[str]] = ["name", "subject"]
    readonly_fields: ClassVar[list[str]] = [
        "recipients_count",
        "opened_count",
        "clicked_count",
        "sent_at",
        "created_at",
    ]

    def status_badge(self, obj: EmailCampaign) -> SafeString:
        """Display status as colored badge."""
        colors = {
            "draft": "gray",
            "scheduled": "orange",
            "sending": "blue",
            "sent": "green",
            "failed": "red",
        }
        color = colors.get(obj.status, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    """Admin interface for notifications."""

    list_display: ClassVar[list[str]] = ["title", "type_badge", "user_id", "is_read", "sent_at"]
    list_filter: ClassVar[list[str]] = ["type", "is_read", "sent_at"]
    search_fields: ClassVar[list[str]] = ["title", "message"]

    def type_badge(self, obj: Notification) -> SafeString:
        """Display notification type as colored badge."""
        colors = {"info": "blue", "warning": "orange", "error": "red", "success": "green"}
        color = colors.get(obj.type, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_type_display(),
        )

    type_badge.short_description = "Type"  # type: ignore[attr-defined]
