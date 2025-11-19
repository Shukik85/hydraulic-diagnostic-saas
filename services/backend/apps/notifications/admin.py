"""Notifications admin with Friendly UX."""

from __future__ import annotations

from typing import ClassVar

from django.contrib import admin
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import EmailCampaign, Notification


@admin.register(EmailCampaign)
class EmailCampaignAdmin(admin.ModelAdmin):
    """Admin interface for email campaigns with Friendly UX."""

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
        """Display status with FriendlyUX badge and icon."""
        badge_classes = {
            "draft": "BadgeMuted",
            "scheduled": "BadgeWarning",
            "sending": "BadgeInfo",
            "sent": "BadgeSuccess",
            "failed": "BadgeError",
        }
        badge_class = badge_classes.get(obj.status, "BadgeMuted")

        icon_name = {
            "draft": "icon-edit",
            "scheduled": "icon-clock",
            "sending": "icon-refresh",
            "sent": "icon-check",
            "failed": "icon-x",
        }.get(obj.status, "icon-minus")

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

    status_badge.short_description = "Status"  # type: ignore[attr-defined]


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    """Admin interface for notifications with Friendly UX."""

    list_display: ClassVar[list[str]] = [
        "title",
        "type_badge",
        "user_id",
        "read_badge",
        "sent_at",
    ]
    list_filter: ClassVar[list[str]] = ["type", "is_read", "sent_at"]
    search_fields: ClassVar[list[str]] = ["title", "message"]

    def type_badge(self, obj: Notification) -> SafeString:
        """Display notification type with FriendlyUX badge and icon."""
        badge_classes = {
            "info": "BadgeInfo",
            "warning": "BadgeWarning",
            "error": "BadgeError",
            "success": "BadgeSuccess",
        }
        badge_class = badge_classes.get(obj.type, "BadgeMuted")

        icon_name = {
            "info": "icon-info",
            "warning": "icon-alert",
            "error": "icon-x",
            "success": "icon-check",
        }.get(obj.type, "icon-minus")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_type_display(),
        )

    type_badge.short_description = "Type"  # type: ignore[attr-defined]

    def read_badge(self, obj: Notification) -> SafeString:
        """Display read status with FriendlyUX badge and icon."""
        if obj.is_read:
            return format_html(
                '<span class="Badge BadgeSuccess">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
                '<use href="{}#icon-check"></use></svg> '
                "Read"
                "</span>",
                static("admin/icons/icons-sprite.svg"),
            )
        return format_html(
            '<span class="Badge BadgeWarning">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#icon-circle"></use></svg> '
            "Unread"
            "</span>",
            static("admin/icons/icons-sprite.svg"),
        )

    read_badge.short_description = "Read Status"  # type: ignore[attr-defined]
