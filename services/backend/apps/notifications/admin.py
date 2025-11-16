"""
Notifications admin
"""

from django.contrib import admin
from django.utils.html import format_html

from .models import EmailCampaign, Notification


@admin.register(EmailCampaign)
class EmailCampaignAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "subject",
        "status_badge",
        "recipients_count",
        "opened_count",
        "scheduled_for",
        "sent_at",
    ]
    list_filter = ["status", "target_tier"]
    search_fields = ["name", "subject"]
    readonly_fields = ["recipients_count", "opened_count", "clicked_count", "sent_at", "created_at"]

    def status_badge(self, obj):
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

    status_badge.short_description = "Status"


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ["title", "type_badge", "user_id", "is_read", "sent_at"]
    list_filter = ["type", "is_read", "sent_at"]
    search_fields = ["title", "message"]

    def type_badge(self, obj):
        colors = {"info": "blue", "warning": "orange", "error": "red", "success": "green"}
        color = colors.get(obj.type, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_type_display(),
        )

    type_badge.short_description = "Type"
