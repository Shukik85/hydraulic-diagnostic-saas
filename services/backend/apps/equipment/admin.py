"""Equipment admin (read-only view) with Friendly UX."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.http import HttpRequest
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import Equipment

if TYPE_CHECKING:
    pass


@admin.register(Equipment)
class EquipmentAdmin(admin.ModelAdmin):
    """Equipment admin interface with Friendly UX badges.

    Read-only view (managed by FastAPI).
    """

    list_display: ClassVar = [
        "system_id",
        "name",
        "system_type_badge",
        "user_id",
        "status_badge",
        "created_at",
    ]
    list_filter: ClassVar = ["system_type", "is_active"]
    search_fields: ClassVar = ["system_id", "name"]
    readonly_fields: ClassVar = [
        "id",
        "user_id",
        "system_id",
        "system_type",
        "name",
        "adjacency_matrix",
        "components",
        "is_active",
        "created_at",
        "updated_at",
    ]

    def system_type_badge(self, obj: Equipment) -> SafeString:
        """Display system type with FriendlyUX badge."""
        badge_classes = {
            "hydraulic": "BadgeInfo",
            "pneumatic": "BadgeWarning",
            "mechanical": "BadgeMuted",
        }
        badge_class = badge_classes.get(obj.system_type, "BadgeMuted")
        return format_html(
            '<span class="Badge {}">{}</span>',
            badge_class,
            obj.system_type.title(),
        )

    system_type_badge.short_description = "System Type"  # type: ignore[attr-defined]

    def status_badge(self, obj: Equipment) -> SafeString:
        """Display active status with FriendlyUX badge and icon."""
        if obj.is_active:
            return format_html(
                '<span class="Badge BadgeSuccess">'
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
                '<use href="{}#icon-check"></use></svg> '
                "Active"
                "</span>",
                static("admin/icons/icons-sprite.svg"),
            )
        return format_html(
            '<span class="Badge BadgeMuted">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#icon-x"></use></svg> '
            "Inactive"
            "</span>",
            static("admin/icons/icons-sprite.svg"),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def has_add_permission(self, request: HttpRequest) -> bool:  # noqa: ARG002
        """Disable add (managed by FastAPI only)."""
        return False

    def has_delete_permission(self, request: HttpRequest, obj: Equipment | None = None) -> bool:  # noqa: ARG002
        """Disable delete (managed by FastAPI only)."""
        return False
