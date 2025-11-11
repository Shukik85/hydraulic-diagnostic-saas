"""
Monitoring admin
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import APILog, ErrorLog


@admin.register(APILog)
class APILogAdmin(admin.ModelAdmin):
    list_display = [
        "method",
        "path_short",
        "status_code_colored",
        "response_time_ms",
        "user_id",
        "created_at",
    ]
    list_filter = ["method", "status_code", "created_at"]
    search_fields = ["path", "user_id"]
    readonly_fields = [
        "id",
        "user_id",
        "method",
        "path",
        "status_code",
        "response_time_ms",
        "ip_address",
        "user_agent",
        "created_at",
    ]

    def has_add_permission(self, request):
        return False

    def path_short(self, obj):
        return f"{obj.path[:50]}..." if len(obj.path) > 50 else obj.path

    path_short.short_description = "Path"

    def status_code_colored(self, obj):
        color = "green" if obj.status_code < 400 else "red"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.status_code,
        )

    status_code_colored.short_description = "Status"


@admin.register(ErrorLog)
class ErrorLogAdmin(admin.ModelAdmin):
    list_display = [
        "severity_badge",
        "error_type",
        "message_short",
        "user_id",
        "created_at",
    ]
    list_filter = ["severity", "error_type", "created_at"]
    search_fields = ["error_type", "message"]
    readonly_fields = [
        "id",
        "user_id",
        "severity",
        "error_type",
        "message",
        "stack_trace",
        "context",
        "created_at",
    ]

    def has_add_permission(self, request):
        return False

    def severity_badge(self, obj):
        colors = {
            "low": "blue",
            "medium": "orange",
            "high": "red",
            "critical": "darkred",
        }
        color = colors.get(obj.severity, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_severity_display(),
        )

    severity_badge.short_description = "Severity"

    def message_short(self, obj):
        return f"{obj.message[:100]}..." if len(obj.message) > 100 else obj.message

    message_short.short_description = "Message"
