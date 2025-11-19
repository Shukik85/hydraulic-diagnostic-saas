"""Monitoring and logging models with Python 3.14 type hints.

Fully typed models for API request logging and error tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from django.db import models

if TYPE_CHECKING:
    pass


class APILog(models.Model):
    """API request log entry.

    Tracks all incoming API requests with timing and metadata.
    Useful for:
    - Performance monitoring
    - Usage analytics
    - Rate limiting decisions
    - Debugging production issues

    Attributes:
        id: UUID primary key
        user_id: Foreign key to User (nullable for anonymous requests)
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        path: Request URL path
        status_code: HTTP response status code
        response_time_ms: Request processing time in milliseconds
        ip_address: Client IP address (v4 or v6)
        user_agent: User-Agent header (truncated to 512 chars)
        created_at: Timestamp of log creation
    """

    id: uuid.UUID = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    user_id: uuid.UUID | None = models.UUIDField(
        db_index=True,
        null=True,
        blank=True,
        help_text="User who made the request (null for anonymous)",
    )

    method: str = models.CharField(
        max_length=10,
        help_text="HTTP method (GET, POST, etc.)",
    )

    path: str = models.CharField(
        max_length=512,
        db_index=True,
        help_text="Request URL path",
    )

    status_code: int = models.IntegerField(
        db_index=True,
        help_text="HTTP response status code",
    )

    response_time_ms: int = models.IntegerField(
        help_text="Response time in milliseconds",
    )

    ip_address: str = models.GenericIPAddressField(
        help_text="Client IP address (IPv4 or IPv6)",
    )

    user_agent: str = models.TextField(
        blank=True,
        default="",
        help_text="User-Agent header (truncated)",
    )

    created_at: datetime = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Log entry timestamp",
    )

    # Type hints for class-level attributes
    if TYPE_CHECKING:
        DoesNotExist: ClassVar[type[Exception]]
        MultipleObjectsReturned: ClassVar[type[Exception]]
        objects: ClassVar[models.Manager[APILog]]

    class Meta:
        db_table = "api_logs"
        verbose_name = "API Log"
        verbose_name_plural = "API Logs"
        ordering: ClassVar[list[str]] = ["-created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["-created_at", "user_id"], name="api_log_user_time_idx"),
            models.Index(fields=["path", "-created_at"], name="api_log_path_time_idx"),
            models.Index(fields=["status_code"], name="api_log_status_idx"),
        ]
        # Partition by month for large tables (optional)
        # managed = False  # If using PostgreSQL partitioning

    def __str__(self) -> str:
        """String representation.

        Returns:
            Human-readable log entry summary
        """
        return f"{self.method} {self.path} - {self.status_code} ({self.response_time_ms}ms)"

    def __repr__(self) -> str:
        """Developer-friendly representation.

        Returns:
            Detailed log entry for debugging
        """
        return (
            f"<APILog id={self.id} method={self.method!r} "
            f"path={self.path!r} status={self.status_code}>"
        )


class ErrorLog(models.Model):
    """Application error log entry.

    Tracks errors, exceptions, and warnings across the application.
    Integrated with Sentry but also stores locally for:
    - Offline analysis
    - Long-term retention
    - Custom aggregation

    Attributes:
        id: UUID primary key
        user_id: User associated with error (if any)
        severity: Error severity level
        error_type: Exception class name or error category
        message: Error message or summary
        stack_trace: Full exception traceback
        context: Additional context as JSON (request data, state, etc.)
        created_at: Timestamp of error occurrence
    """

    # Severity level choices
    class Severity(models.TextChoices):
        """Error severity levels."""

        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"
        CRITICAL = "critical", "Critical"

    id: uuid.UUID = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    user_id: uuid.UUID | None = models.UUIDField(
        db_index=True,
        null=True,
        blank=True,
        help_text="User associated with error (if applicable)",
    )

    severity: str = models.CharField(
        max_length=20,
        choices=Severity.choices,
        default=Severity.MEDIUM,
        db_index=True,
        help_text="Error severity level",
    )

    error_type: str = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Exception type or error category",
    )

    message: str = models.TextField(
        help_text="Error message or summary",
    )

    stack_trace: str = models.TextField(
        blank=True,
        default="",
        help_text="Full exception traceback",
    )

    context: dict[str, Any] = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional context (request data, user state, etc.)",
    )

    created_at: datetime = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Error occurrence timestamp",
    )

    # Type hints for class-level attributes
    if TYPE_CHECKING:
        DoesNotExist: ClassVar[type[Exception]]
        MultipleObjectsReturned: ClassVar[type[Exception]]
        objects: ClassVar[models.Manager[ErrorLog]]

    class Meta:
        db_table = "error_logs"
        verbose_name = "Error Log"
        verbose_name_plural = "Error Logs"
        ordering: ClassVar[list[str]] = ["-created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["-created_at", "severity"], name="error_log_sev_time_idx"),
            models.Index(fields=["error_type"], name="error_log_type_idx"),
        ]

    def __str__(self) -> str:
        """String representation.

        Returns:
            Human-readable error summary
        """
        return f"[{self.severity.upper()}] {self.error_type}: {self.message[:100]}"

    def __repr__(self) -> str:
        """Developer-friendly representation.

        Returns:
            Detailed error info for debugging
        """
        return f"<ErrorLog id={self.id} severity={self.severity!r} type={self.error_type!r}>"

    @property
    def severity_enum(self) -> Severity:
        """Get severity as enum for type-safe comparisons.

        Returns:
            Severity enum value
        """
        return self.Severity(self.severity)

    @property
    def is_critical(self) -> bool:
        """Check if error is critical severity.

        Returns:
            True if severity is CRITICAL
        """
        return self.severity_enum == self.Severity.CRITICAL
