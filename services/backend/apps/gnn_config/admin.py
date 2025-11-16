"""Расширенный Django Admin: экшены для деплоя, rollback и live sync c FastAPI GNN service"""

from typing import ClassVar

from django.conf import settings
from django.contrib import admin, messages
from django.utils.html import format_html

from .gnn_service_client import GNNAdminClient
from .models import GNNModelConfig, GNNTrainingJob


@admin.register(GNNModelConfig)
class GNNModelConfigAdmin(admin.ModelAdmin):
    list_display: ClassVar = (
        "model_version",
        "is_active",
        "deployed_at",
        "framework",
        "size_mb",
        "input_shape",
        "output_classes",
        "deployed_by",
        "prod_status",
    )
    list_filter: ClassVar = ("is_active", "framework", "deployed_at")
    search_fields: ClassVar = ("model_version", "description")
    readonly_fields: ClassVar = ("deployed_at", "deployed_by", "size_mb", "prod_sync_status")
    ordering: ClassVar = ("-is_active", "-deployed_at")
    actions: ClassVar = ["deploy_to_production", "sync_from_service"]
    fieldsets = (
        (
            "Model Info",
            {
                "fields": (
                    "model_version",
                    "is_active",
                    "framework",
                    "path",
                    "input_shape",
                    "output_classes",
                    "size_mb",
                    "deployed_at",
                    "deployed_by",
                    "prod_sync_status",
                )
            },
        ),
        ("Описание", {"fields": ("description",)}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GNNAdminClient(
            base_url=getattr(settings, "GNN_SERVICE_URL", "http://gnn-service:8002"),
            admin_token=getattr(settings, "GNN_ADMIN_TOKEN", None),
        )

    def prod_status(self, obj: GNNModelConfig):
        if obj.is_active:
            return format_html('<b style="color: #16a34a;">Active</b>')
        return "-"

    prod_status.short_description = "Prod"

    def prod_sync_status(self, obj: GNNModelConfig):
        try:
            info = self.client.get_model_info()
            if info["model_version"] == obj.model_version:
                return format_html('<span style="color: #16a34a;">Production version</span>')
            return format_html(
                '<span style="color: #eab308;">Out of sync (v{})</span>', info["model_version"]
            )
        except Exception as e:
            return format_html('<span style="color: #dc2626;">Sync error ({})</span>', str(e)[:50])

    prod_sync_status.short_description = "Prod Sync"

    @admin.action(description="Deploy to production")
    def deploy_to_production(self, request, queryset):
        for obj in queryset:
            try:
                res = self.client.deploy_model(
                    model_path=obj.path,
                    version=obj.model_version,
                    description=obj.description,
                    validate_first=True,
                )
                obj.is_active = True
                obj.save()
                self.message_user(
                    request,
                    f"Model {obj.model_version} deployed: {res.get('message')}",
                    messages.SUCCESS,
                )
            except Exception as e:
                self.message_user(request, f"Failed: {e!s}", messages.ERROR)

    @admin.action(description="Sync model passport from GNN service")
    def sync_from_service(self, request, queryset):
        """Update Django model metadata from GNN service."""
        try:
            info = self.client.get_model_info()
            fields: ClassVar = [
                "model_version",
                "framework",
                "input_shape",
                "output_classes",
                "deployed_at",
                "size_mb",
                "model_path",
            ]
            if not queryset.exists():
                self.message_user(request, "No selected models for sync!", messages.WARNING)
                return
            for obj in queryset:
                for field in fields:
                    val = info.get(field)
                    if val is not None:
                        setattr(obj, field if field != "model_path" else "path", val)
                obj.save()
                self.message_user(
                    request, f"Updated {obj.model_version} from prod", messages.SUCCESS
                )
        except Exception as e:
            self.message_user(request, f"Sync failed: {e!s}", messages.ERROR)

    def has_delete_permission(self, request, obj=None):
        if obj and obj.is_active:
            return False
        return super().has_delete_permission(request, obj)


@admin.register(GNNTrainingJob)
class GNNTrainingJobAdmin(admin.ModelAdmin):
    list_display: ClassVar = (
        "job_id",
        "status",
        "experiment_name",
        "started_by",
        "started_at",
        "finished_at",
        "tensorboard_url",
    )
    list_filter: ClassVar = ("status", "started_at", "started_by")
    search_fields: ClassVar = ("job_id", "experiment_name", "dataset_path")
    readonly_fields: ClassVar = ("started_at", "finished_at", "tensorboard_url")
    ordering: ClassVar = ("-started_at",)
    actions: ClassVar = ["trigger_training", "refresh_status"]
    fieldsets = (
        (
            "Overview",
            {
                "fields": (
                    "job_id",
                    "status",
                    "experiment_name",
                    "dataset_path",
                    "started_by",
                    "started_at",
                    "finished_at",
                    "tensorboard_url",
                )
            },
        ),
        ("Config & Metrics", {"fields": ("config", "last_metrics")}),
        ("Notes", {"fields": ("notes",)}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GNNAdminClient(
            base_url=getattr(settings, "GNN_SERVICE_URL", "http://gnn-service:8002"),
            admin_token=getattr(settings, "GNN_ADMIN_TOKEN", None),
        )

    @admin.action(description="Trigger training via GNN service API")
    def trigger_training(self, request, queryset):
        for obj in queryset:
            try:
                res = self.client.start_training(
                    dataset_path=obj.dataset_path,
                    config=obj.config,
                    experiment_name=obj.experiment_name,
                )
                obj.job_id = res.get("job_id", obj.job_id)
                obj.status = res.get("status", obj.status)
                obj.tensorboard_url = res.get("tensorboard_url", obj.tensorboard_url)
                obj.save()
                self.message_user(request, f"Training triggered: {obj.job_id}", messages.SUCCESS)
            except Exception as e:
                self.message_user(request, f"Failed: {e!s}", messages.ERROR)

    @admin.action(description="Refresh status from GNN service")
    def refresh_status(self, request, queryset):
        for obj in queryset:
            try:
                res = self.client.get_training_status(obj.job_id)
                obj.status = res.get("status", obj.status)
                obj.last_metrics = res
                obj.save()
                self.message_user(
                    request, f"Status updated: {obj.job_id}={obj.status}", messages.SUCCESS
                )
            except Exception as e:
                self.message_user(request, f"Status update failed: {e!s}", messages.ERROR)

    def has_delete_permission(self, request, obj=None):
        return False
