"""
Quarantine models for invalid sensor readings.

Enterprise data quality management with full audit trail.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, ClassVar

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


class QuarantinedReadingQuerySet(models.QuerySet["QuarantinedReading"]):
    """Custom QuerySet for QuarantinedReading."""
    
    def for_job(self, job_id: uuid.UUID) -> "QuarantinedReadingQuerySet":
        """Filter by ingestion job."""
        return self.filter(job_id=job_id)
    
    def by_reason(self, reason: str) -> "QuarantinedReadingQuerySet":
        """Filter by quarantine reason."""
        return self.filter(reason=reason)
    
    def pending_review(self) -> "QuarantinedReadingQuerySet":
        """Get readings pending manual review."""
        return self.filter(review_status='pending')


class QuarantinedReading(models.Model):
    """
    Quarantined sensor readings that failed validation.
    
    Used for data quality tracking and manual review workflow.
    Designed for TimescaleDB with BRIN indexes on timestamp.
    """
    
    REASON_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('out_of_range', 'Value out of valid range'),
        ('invalid_timestamp', 'Invalid or future timestamp'),
        ('duplicate', 'Duplicate reading detected'),
        ('parse_error', 'Failed to parse reading'),
        ('system_not_found', 'System ID not found'),
        ('invalid_unit', 'Invalid measurement unit'),
        ('other', 'Other validation error'),
    ]
    
    REVIEW_STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved - Will Retry'),
        ('rejected', 'Rejected - Discard'),
        ('fixed', 'Fixed and Reprocessed'),
    ]
    
    id: models.UUIDField = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    
    # Reference to ingestion job
    job_id: models.UUIDField = models.UUIDField(
        db_index=True,
        help_text="UUID of the ingestion job that quarantined this reading"
    )
    
    # Original sensor reading data
    sensor_id: models.UUIDField = models.UUIDField(
        help_text="Original sensor ID from reading"
    )
    
    timestamp: models.DateTimeField = models.DateTimeField(
        db_index=True,
        help_text="Original timestamp from reading"
    )
    
    value: models.FloatField = models.FloatField(
        help_text="Original sensor value"
    )
    
    unit: models.CharField = models.CharField(
        max_length=32,
        help_text="Original unit of measurement"
    )
    
    quality: models.IntegerField = models.IntegerField(
        default=0,
        help_text="Original quality score (0-100)"
    )
    
    # System reference (may be null if system not found)
    system_id: models.UUIDField = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="System ID if available"
    )
    
    # Quarantine metadata
    reason: models.CharField = models.CharField(
        max_length=32,
        choices=REASON_CHOICES,
        db_index=True,
        help_text="Reason for quarantine"
    )
    
    reason_details: models.TextField = models.TextField(
        blank=True,
        default="",
        help_text="Detailed explanation of validation failure"
    )
    
    # Review workflow
    review_status: models.CharField = models.CharField(
        max_length=20,
        choices=REVIEW_STATUS_CHOICES,
        default='pending',
        db_index=True
    )
    
    reviewed_by: models.ForeignKey = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_quarantined_readings',
        help_text="User who reviewed this reading"
    )
    
    reviewed_at: models.DateTimeField = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When review was completed"
    )
    
    review_notes: models.TextField = models.TextField(
        blank=True,
        default="",
        help_text="Notes from manual review"
    )
    
    # Timestamps
    quarantined_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now,
        db_index=True
    )
    
    objects = models.Manager()
    qs: "QuarantinedReadingQuerySet" = QuarantinedReadingQuerySet.as_manager()  # type: ignore[assignment]
    
    class Meta:
        db_table = "diagnostics_quarantined_reading"
        ordering = ["-quarantined_at"]
        
        indexes = [
            # Primary queries
            BTreeIndex(fields=["job_id", "quarantined_at"], name="idx_qr_job_time"),
            BTreeIndex(fields=["reason", "quarantined_at"], name="idx_qr_reason_time"),
            BTreeIndex(fields=["review_status", "quarantined_at"], name="idx_qr_status_time"),
            
            # BRIN for timestamp (TimescaleDB optimized)
            BrinIndex(fields=["quarantined_at"], autosummarize=True, name="brin_qr_time"),
        ]
        
        verbose_name = "Quarantined Reading"
        verbose_name_plural = "Quarantined Readings"
    
    def __str__(self) -> str:
        return f"Quarantined: sensor={self.sensor_id} reason={self.reason} job={self.job_id}"
    
    def __repr__(self) -> str:
        return f"<QuarantinedReading id={self.id} reason={self.reason} status={self.review_status}>"


__all__ = [
    'QuarantinedReading',
    'QuarantinedReadingQuerySet',
]
