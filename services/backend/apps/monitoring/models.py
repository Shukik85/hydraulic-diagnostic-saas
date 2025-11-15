"""
Monitoring and logging models
"""
import uuid

from django.db import models


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
        db_table = 'api_logs'
        verbose_name = 'API Log'
        verbose_name_plural = 'API Logs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at', 'user_id']),
        ]

    def __str__(self):
        return f"{self.method} {self.path} - {self.status_code}"


class ErrorLog(models.Model):
    """Error tracking"""

    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True, null=True, blank=True)

    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    error_type = models.CharField(max_length=255)
    message = models.TextField()
    stack_trace = models.TextField(blank=True)

    context = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'error_logs'
        verbose_name = 'Error Log'
        verbose_name_plural = 'Error Logs'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.severity}] {self.error_type}"
