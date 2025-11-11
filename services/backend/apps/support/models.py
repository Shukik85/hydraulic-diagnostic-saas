"""
Support ticket model (synced with FastAPI DB)
"""
from django.db import models
import uuid


class SupportTicket(models.Model):
    """
    Support tickets (managed by Django Admin, created by FastAPI)
    """

    STATUS_CHOICES = [
        ('open', 'Open'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('closed', 'Closed'),
    ]

    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)

    subject = models.CharField(max_length=255)
    message = models.TextField()
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='medium')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    category = models.CharField(max_length=100, blank=True, null=True)

    assigned_to = models.UUIDField(null=True, blank=True)
    response = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'support_tickets'
        verbose_name = 'Support Ticket'
        verbose_name_plural = 'Support Tickets'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.priority.upper()}] {self.subject}"


class DataExportRequest(models.Model):
    """
    Data export requests (GDPR)
    """

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    download_url = models.URLField(max_length=512, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'data_export_requests'
        verbose_name = 'Data Export Request'
        verbose_name_plural = 'Data Export Requests'
        ordering = ['-created_at']

    def __str__(self):
        return f"Export {self.id} - {self.status}"


class SupportAction(models.Model):
    """
    Support action log (admin actions on user accounts)
    """
    
    ACTION_CHOICES = [
        ('password_reset', 'Password Reset'),
        ('trial_extend', 'Trial Extended'),
        ('subscription_change', 'Subscription Changed'),
        ('account_unlock', 'Account Unlocked'),
        ('data_recovery', 'Data Recovery'),
        ('other', 'Other'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)
    action_type = models.CharField(max_length=100, choices=ACTION_CHOICES)
    performed_by = models.UUIDField()  # Admin user ID
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'support_actions'
        verbose_name = 'Support Action'
        verbose_name_plural = 'Support Actions'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.action_type} for {self.user_id}"
