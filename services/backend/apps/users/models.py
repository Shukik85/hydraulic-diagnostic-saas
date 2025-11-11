"""
User models with subscription management
"""
from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid

class User(AbstractUser):
    """Custom User model with subscription and API key"""

    TIER_CHOICES = [
        ('free', 'Free'),
        ('pro', 'Pro'),
        ('enterprise', 'Enterprise'),
    ]

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('trial', 'Trial'),
        ('cancelled', 'Cancelled'),
        ('expired', 'Expired'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    api_key = models.CharField(max_length=128, unique=True, blank=True)

    # Subscription
    subscription_tier = models.CharField(max_length=20, choices=TIER_CHOICES, default='free')
    subscription_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='trial')
    trial_end_date = models.DateTimeField(null=True, blank=True)

    # Billing
    stripe_customer_id = models.CharField(max_length=255, blank=True, null=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True)

    # Usage tracking
    api_requests_count = models.IntegerField(default=0)
    ml_inferences_count = models.IntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']

    def __str__(self):
        return self.email

    def save(self, *args, **kwargs):
        if not self.api_key:
            import secrets
            self.api_key = f"hyd_{secrets.token_urlsafe(32)}"
        super().save(*args, **kwargs)
