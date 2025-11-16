"""GNN Configuration & Training management models for orchestration (Python 3.14)"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.db import models

if TYPE_CHECKING:
    from apps.users.models import User


class GNNModelConfig(models.Model):
    """Production/registry passport for deployed GNN models."""

    id: uuid.UUID = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_version: str = models.CharField(max_length=30, db_index=True)
    deployed_at: datetime = models.DateTimeField(auto_now_add=True)
    path: str = models.CharField(
        max_length=255, help_text="Filesystem or artifact path to model file"
    )
    size_mb: float = models.FloatField(help_text="Model size in MB", default=0)
    framework: str = models.CharField(max_length=32, help_text="onnx, pytorch, etc.")
    input_shape: str = models.CharField(
        max_length=64, help_text="Comma-separated shape (e.g., 1,10,32)"
    )
    output_classes: int = models.PositiveSmallIntegerField(default=3)
    deployed_by: User | None = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    description: str = models.TextField(blank=True, default="")
    is_active: bool = models.BooleanField(
        default=False, help_text="Is this model currently in production?"
    )

    class Meta:
        db_table = "gnn_model_config"
        verbose_name = "GNN Model Configuration"
        verbose_name_plural = "GNN Model Configurations"
        ordering: ClassVar[list[str]] = ["-deployed_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["model_version", "is_active"], name="model_version_active_idx")
        ]

    def __str__(self) -> str:
        return f"GNNModel v{self.model_version} ({'active' if self.is_active else 'archived'})"


class GNNTrainingJob(models.Model):
    """Training job history for GNN experiments."""

    id: uuid.UUID = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job_id: str = models.CharField(max_length=40, unique=True, db_index=True)
    started_by: User | None = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    status: str = models.CharField(
        max_length=16,
        choices=[
            ("queued", "Queued"),
            ("started", "Started"),
            ("running", "Running"),
            ("done", "Done"),
            ("failed", "Failed"),
            ("stopped", "Stopped"),
        ],
        default="queued",
        db_index=True,
    )
    started_at: datetime = models.DateTimeField(auto_now_add=True)
    finished_at: datetime | None = models.DateTimeField(null=True, blank=True)
    config: dict[str, Any] = models.JSONField(default=dict)
    dataset_path: str = models.CharField(max_length=255)
    experiment_name: str = models.CharField(max_length=128)
    tensorboard_url: str = models.CharField(max_length=255, blank=True, default="")
    last_metrics: dict[str, Any] = models.JSONField(default=dict, blank=True)
    notes: str = models.TextField(blank=True, default="")

    class Meta:
        db_table = "gnn_training_job"
        verbose_name = "GNN Training Job"
        verbose_name_plural = "GNN Training Jobs"
        ordering: ClassVar[list[str]] = ["-started_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["job_id", "status"], name="jobid_status_idx")
        ]

    def __str__(self) -> str:
        return f"GNN TrainJob {self.job_id} ({self.status})"
