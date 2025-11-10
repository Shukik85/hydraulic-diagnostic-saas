"""
Ingestion job tracking models for sensor data pipeline.

Enterprise job status tracking with full observability.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, ClassVar

from django.contrib.postgres.indexes import BTreeIndex
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


class IngestionJobQuerySet(models.QuerySet["IngestionJob"]):
    """Custom QuerySet for IngestionJob."""
    
    def active(self) -> "IngestionJobQuerySet":
        """Get jobs that are queued or processing."""
        return self.filter(status__in=['queued', 'processing'])
    
    def completed(self) -> "IngestionJobQuerySet":
        """Get completed jobs (success or failure)."""
        return self.filter(status__in=['completed', 'failed'])
    
    def by_status(self, status: str) -> "IngestionJobQuerySet":
        """Filter by job status."""
        return self.filter(status=status)


class IngestionJob(models.Model):
    """
    Sensor data ingestion job tracking.
    
    Tracks bulk sensor data ingestion lifecycle:
    - Queued: Job accepted, waiting for processing
    - Processing: Celery task is running
    - Completed: Successfully finished
    - Failed: Error during processing
    
    Provides full observability for data pipeline monitoring.
    """
    
    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('queued', 'Queued'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id: models.UUIDField = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique job identifier"
    )
    
    # Job status and timing
    status: models.CharField = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='queued',
        db_index=True,
        help_text="Current job status"
    )
    
    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When job was created"
    )
    
    started_at: models.DateTimeField = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When processing started"
    )
    
    completed_at: models.DateTimeField = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When job finished (success or failure)"
    )
    
    # Data counters
    total_readings: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0,
        help_text="Total number of readings in batch"
    )
    
    inserted_readings: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0,
        help_text="Successfully inserted readings"
    )
    
    quarantined_readings: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0,
        help_text="Readings quarantined due to validation errors"
    )
    
    # System reference
    system_id: models.UUIDField = models.UUIDField(
        db_index=True,
        help_text="Target hydraulic system ID"
    )
    
    # Error tracking
    error_message: models.TextField = models.TextField(
        blank=True,
        default="",
        help_text="Error message if job failed"
    )
    
    # Performance metrics
    processing_time_ms: models.PositiveIntegerField = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Total processing time in milliseconds"
    )
    
    # Celery task reference
    celery_task_id: models.CharField = models.CharField(
        max_length=255,
        blank=True,
        default="",
        db_index=True,
        help_text="Celery task ID for tracking"
    )
    
    # User who initiated the job
    created_by: models.ForeignKey = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='ingestion_jobs',
        help_text="User who initiated ingestion"
    )
    
    objects = models.Manager()
    qs: "IngestionJobQuerySet" = IngestionJobQuerySet.as_manager()  # type: ignore[assignment]
    
    class Meta:
        db_table = "diagnostics_ingestion_job"
        ordering = ["-created_at"]
        
        indexes = [
            BTreeIndex(fields=["status", "created_at"], name="idx_ij_status_created"),
            BTreeIndex(fields=["system_id", "created_at"], name="idx_ij_system_created"),
            BTreeIndex(fields=["celery_task_id"], name="idx_ij_celery_task"),
        ]
        
        verbose_name = "Ingestion Job"
        verbose_name_plural = "Ingestion Jobs"
    
    def __str__(self) -> str:
        return f"IngestionJob {self.id}: {self.status} ({self.inserted_readings}/{self.total_readings})"
    
    def __repr__(self) -> str:
        return f"<IngestionJob id={self.id} status={self.status} system={self.system_id}>"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_readings == 0:
            return 0.0
        return (self.inserted_readings / self.total_readings) * 100.0
    
    @property
    def is_active(self) -> bool:
        """Check if job is still running."""
        return self.status in ['queued', 'processing']
    
    @property
    def is_completed(self) -> bool:
        """Check if job is finished (success or failure)."""
        return self.status in ['completed', 'failed']


__all__ = [
    'IngestionJob',
    'IngestionJobQuerySet',
]
