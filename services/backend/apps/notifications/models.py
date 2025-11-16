"""
Refactored EmailCampaign and Notification models to use shared enums. Added property for status/type.
"""

import uuid

from django.db import models

from apps.core.enums import EmailCampaignStatus, NotificationType


class EmailCampaign(models.Model):
    """Email marketing campaigns"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    subject = models.CharField(max_length=255)
    body = models.TextField()
    target_tier = models.CharField(max_length=20, blank=True)  # Enum enforced via API
    target_status = models.CharField(max_length=20, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[(s.value, s.name.title()) for s in EmailCampaignStatus],
        default=EmailCampaignStatus.DRAFT.value,
    )
    scheduled_for = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    recipients_count = models.IntegerField(default=0)
    opened_count = models.IntegerField(default=0)
    clicked_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "email_campaigns"
        verbose_name = "Email Campaign"
        verbose_name_plural = "Email Campaigns"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def status_enum(self) -> EmailCampaignStatus:
        return EmailCampaignStatus(self.status)


class Notification(models.Model):
    """System notifications"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True, null=True, blank=True)  # Null = all users
    type = models.CharField(
        max_length=20,
        choices=[(n.value, n.name.title()) for n in NotificationType],
    )
    title = models.CharField(max_length=255)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    sent_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "notifications"
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"
        ordering = ["-sent_at"]

    def __str__(self):
        return f"{self.type}: {self.title}"

    @property
    def type_enum(self) -> NotificationType:
        return NotificationType(self.type)
