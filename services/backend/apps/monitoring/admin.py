"""Monitoring admin with type-safe class attributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import APILog, ErrorLog

if TYPE_CHECKING:
    from django.http import HttpRequest


@admin.register(APILog)
class APILogAdmin(admin.ModelAdmin):
    """Admin interface for API logs."""

    list_display: ClassVar[list[str]] = [
        "method",
        "path_short",
        "status_code_colored",
        "response_time_ms",
        "user_id",
        "created_at",
    ]
    list_filter: ClassVar[list[str]] = ["method", "status_code", "created_at"]
    search_fields: ClassVar[list[str]] = ["path", "user_id"]
    readonly_fields: ClassVar[list[str]] = [
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

    def has_add_permission(self, request: HttpRequest) -> bool:  # noqa: ARG002
        """Disable manual log creation."""
        return False

    def path_short(self, obj: APILog) -> str:
        """Truncate long paths."""
        return f"{obj.path[:50]}..." if len(obj.path) > 50 else obj.path

    path_short.short_description = "Path"  # type: ignore[attr-defined]

    def status_code_colored(self, obj: APILog) -> SafeString:
        """Display status code with color."""
        color = "green" if obj.status_code < 400 else "red"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.status_code,
        )

    status_code_colored.short_description = "Status"  # type: ignore[attr-defined]


@admin.register(ErrorLog)
class ErrorLogAdmin(admin.ModelAdmin):
    """Admin interface for error logs."""

    list_display: ClassVar[list[str]] = [
        "severity_badge",
        "error_type",
        "message_short",
        "user_id",
        "created_at",
    ]
    list_filter: ClassVar[list[str]] = ["severity", "error_type", "created_at"]
    search_fields: ClassVar[list[str]] = ["error_type", "message"]
    readonly_fields: ClassVar[list[str]] = [
        "id",
        "user_id",
        "severity",
        "error_type",
        "message",
        "stack_trace",
        "context",
        "created_at",
    ]

    def has_add_permission(self, request: HttpRequest) -> bool:  # noqa: ARG002
        """Disable manual log creation."""
        return False

    def severity_badge(self, obj: ErrorLog) -> SafeString:
        """Display severity as colored badge."""
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

    severity_badge.short_description = "Severity"  # type: ignore[attr-defined]

    def message_short(self, obj: ErrorLog) -> str:
        """Truncate long messages."""
        return f"{obj.message[:100]}..." if len(obj.message) > 100 else obj.message

    message_short.short_description = "Message"  # type: ignore[attr-defined]
