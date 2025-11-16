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
        "created_at",
    ]
    list_filter: ClassVar = ["system_type"]
    search_fields: ClassVar = ["system_id", "name"]
    readonly_fields: ClassVar = [
        "id",
        "user_id",
        "system_id",
        "system_type",
        "name",
        "adjacency_matrix",
        "components",
        "created_at",
        "updated_at",
    ]

    def system_type_badge(self, obj: Equipment) -> SafeString:
        """Display system type with FriendlyUX badge."""
        badge_classes = {
            "hydraulic": "BadgeInfo",
            "pneumatic": "BadgeWarning",
            "electrical": "BadgeSuccess",
            "other": "BadgeMuted",
        }
        badge_class = badge_classes.get(obj.system_type, "BadgeMuted")

        icon_name = {
            "hydraulic": "icon-equipment",
            "pneumatic": "icon-wind",
            "electrical": "icon-zap",
            "other": "icon-box",
        }.get(obj.system_type, "icon-box")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_system_type_display(),
        )

    system_type_badge.short_description = "System Type"  # type: ignore[attr-defined]

    def has_add_permission(self, request: HttpRequest) -> bool:  # noqa: ARG002
        """Disable add (managed by FastAPI only)."""
        return False

    def has_delete_permission(self, request: HttpRequest, obj: Equipment | None = None) -> bool:  # noqa: ARG002
        """Disable delete (managed by FastAPI only)."""
        return False
