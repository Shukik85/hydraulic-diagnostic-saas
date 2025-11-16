"""
Equipment admin (read-only view)
"""

from typing import ClassVar

from django.contrib import admin

from .models import Equipment


@admin.register(Equipment)
class EquipmentAdmin(admin.ModelAdmin):
    list_display: ClassVar = ["system_id", "name", "system_type", "user_id", "created_at"]
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

    def has_add_permission(self, request):  # noqa: ARG002
        return False  # Managed by FastAPI only

    def has_delete_permission(self, request, obj=None):  # noqa: ARG002
        return False  # Managed by FastAPI only
