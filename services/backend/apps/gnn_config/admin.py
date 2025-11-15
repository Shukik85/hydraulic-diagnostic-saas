"""Admin для управления GNN моделями и запусками обучения."""
from django.contrib import admin
from .models import GNNModelConfig, GNNTrainingJob

@admin.register(GNNModelConfig)
class GNNModelConfigAdmin(admin.ModelAdmin):
    list_display = (
        "model_version", "is_active", "deployed_at", "framework", "size_mb", "input_shape", "output_classes", "deployed_by"
    )
    list_filter = ("is_active", "framework", "deployed_at")
    search_fields = ("model_version", "description")
    readonly_fields = ("deployed_at", "deployed_by", "size_mb")
    ordering = ("-is_active", "-deployed_at")
    fieldsets = (
        ("Model Info", {
            "fields": ("model_version", "is_active", "framework", "path", "input_shape", "output_classes", "size_mb", "deployed_at", "deployed_by")
        }),
        ("Описание", {"fields": ("description",)})
    )
    def has_delete_permission(self, request, obj=None):
        # Block deletion of active production model
        if obj and obj.is_active:
            return False
        return super().has_delete_permission(request, obj)

@admin.register(GNNTrainingJob)
class GNNTrainingJobAdmin(admin.ModelAdmin):
    list_display = ("job_id", "status", "experiment_name", "started_by", "started_at", "finished_at", "tensorboard_url")
    list_filter = ("status", "started_at", "started_by")
    search_fields = ("job_id", "experiment_name", "dataset_path")
    readonly_fields = ("started_at", "finished_at", "tensorboard_url")
    ordering = ("-started_at",)
    fieldsets = (
        ("Overview", {"fields": ("job_id", "status", "experiment_name", "dataset_path", "started_by", "started_at", "finished_at", "tensorboard_url")}),
        ("Config & Metrics", {"fields": ("config", "last_metrics")}),
        ("Notes", {"fields": ("notes",)})
    )
    def has_delete_permission(self, request, obj=None):
        # Never delete job history via admin
        return False
