"""
Monitoring and support models now use shared business enums for all status, severity, kind fields.
"""

import uuid

from django.db import models

from apps.core.enums import (
    DataExportStatus,
    ErrorSeverity,
    SupportActionType,
    SupportPriority,
    SupportTicketStatus,
)


class APILog(models.Model):
    """API request logging"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True, null=True, blank=True)
    method = models.CharField(max_length=10)
    path = models.CharField(max_length=512)
    status_code = models.IntegerField()
    response_time_ms = models.IntegerField()
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "API Log"
        verbose_name_plural = "API Logs"
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["created_at", "user_id"])]  # порядок только в ordering

    def __str__(self):
        return f"{self.method} {self.path} - {self.status_code}"


class ErrorLog(models.Model):
    """Error tracking"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True, null=True, blank=True)
    severity = models.CharField(
        max_length=20,
        choices=[(s.value, s.name.title()) for s in ErrorSeverity],
    )
    error_type = models.CharField(max_length=255)
    message = models.TextField()
    stack_trace = models.TextField(blank=True)
    context = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "Error Log"
        verbose_name_plural = "Error Logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.severity}] {self.error_type}"

    @property
    def severity_enum(self) -> ErrorSeverity:
        return ErrorSeverity(self.severity)


class SupportTicket(models.Model):
    STATUS_CHOICES = [(s.value, s.name.title()) for s in SupportTicketStatus]
    PRIORITY_CHOICES = [(p.value, p.name.title()) for p in SupportPriority]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)
    subject = models.CharField(max_length=255)
    message = models.TextField()
    priority = models.CharField(
        max_length=20, choices=PRIORITY_CHOICES, default=SupportPriority.MEDIUM.value
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=SupportTicketStatus.OPEN.value
    )
    category = models.CharField(max_length=100, blank=True, null=True)
    assigned_to = models.UUIDField(null=True, blank=True)
    response = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "support_tickets"
        verbose_name = "Support Ticket"
        verbose_name_plural = "Support Tickets"
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.priority.upper()}] {self.subject}"

    @property
    def status_enum(self) -> SupportTicketStatus:
        return SupportTicketStatus(self.status)

    @property
    def priority_enum(self) -> SupportPriority:
        return SupportPriority(self.priority)


class DataExportRequest(models.Model):
    STATUS_CHOICES = [(s.value, s.name.title()) for s in DataExportStatus]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=DataExportStatus.PENDING.value
    )
    download_url = models.URLField(max_length=512, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "data_export_requests"
        verbose_name = "Data Export Request"
        verbose_name_plural = "Data Export Requests"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Export {self.id} - {self.status}"

    @property
    def status_enum(self) -> DataExportStatus:
        return DataExportStatus(self.status)


class SupportAction(models.Model):
    ACTION_CHOICES = [(a.value, a.name.replace("_", " ").title()) for a in SupportActionType]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)
    action_type = models.CharField(max_length=100, choices=ACTION_CHOICES)
    performed_by = models.UUIDField()
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "support_actions"
        verbose_name = "Support Action"
        verbose_name_plural = "Support Actions"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.action_type} for {self.user_id}"

    @property
    def action_enum(self) -> SupportActionType:
        return SupportActionType(self.action_type)
