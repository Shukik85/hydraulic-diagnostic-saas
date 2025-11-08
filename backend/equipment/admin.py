"""Equipment admin"""
from django.contrib import admin
from equipment.models.equipment_config import EquipmentConfig, ConfigExtractionLog


@admin.register(EquipmentConfig)
class EquipmentConfigAdmin(admin.ModelAdmin):
    list_display = [
        "equipment_id",
        "manufacturer",
        "model",
        "status",
        "uploaded_by",
        "created_at",
    ]
    list_filter = ["status", "manufacturer", "deployed"]
    search_fields = ["equipment_id", "manufacturer", "model"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(ConfigExtractionLog)
class ConfigExtractionLogAdmin(admin.ModelAdmin):
    list_display = ["config", "created_at"]
    readonly_fields = ["extracted_data", "confidence_scores", "created_at"]
