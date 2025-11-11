"""
Equipment admin (read-only view)
"""
from django.contrib import admin
from .models import Equipment

@admin.register(Equipment)
class EquipmentAdmin(admin.ModelAdmin):
    list_display = ['system_id', 'name', 'system_type', 'user_id', 'created_at']
    list_filter = ['system_type']
    search_fields = ['system_id', 'name']
    readonly_fields = ['id', 'user_id', 'system_id', 'system_type', 'name', 
                       'adjacency_matrix', 'components', 'created_at', 'updated_at']

    def has_add_permission(self, request):
        return False  # Managed by FastAPI only

    def has_delete_permission(self, request, obj=None):
        return False  # Managed by FastAPI only
