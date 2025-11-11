"""
Notification and email campaign models
"""
from django.db import models
import uuid

class EmailCampaign(models.Model):
    """Email marketing campaigns"""

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('scheduled', 'Scheduled'),
        ('sending', 'Sending'),
        ('sent', 'Sent'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    subject = models.CharField(max_length=255)
    body = models.TextField()

    target_tier = models.CharField(max_length=20, blank=True)  # Empty = all users
    target_status = models.CharField(max_length=20, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    scheduled_for = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)

    recipients_count = models.IntegerField(default=0)
    opened_count = models.IntegerField(default=0)
    clicked_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'email_campaigns'
        verbose_name = 'Email Campaign'
        verbose_name_plural = 'Email Campaigns'
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class Notification(models.Model):
    """System notifications"""

    TYPE_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('success', 'Success'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True, null=True, blank=True)  # Null = all users

    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    title = models.CharField(max_length=255)
    message = models.TextField()

    is_read = models.BooleanField(default=False)
    sent_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        verbose_name = 'Notification'
        verbose_name_plural = 'Notifications'
        ordering = ['-sent_at']

    def __str__(self):
        return f"{self.type}: {self.title}"
