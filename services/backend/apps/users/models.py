"""User models with subscription management.

Refactored to use:
- Python 3.14 type hints (PEP 604 union syntax)
- Shared enums from apps.core.enums
- Proper type annotations
- Atomic API key generation
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from django.contrib.auth.models import AbstractUser
from django.db import IntegrityError, models, transaction
from django.utils import timezone

from apps.core.enums import SubscriptionStatus, SubscriptionTier

if TYPE_CHECKING:
    from django.db.models.manager import RelatedManager


class User(AbstractUser):
    """Custom User model with subscription and API key.

    Attributes:
        id: UUID primary key
        email: Unique email address (used for authentication)
        api_key: Unique API key for external service access
        subscription_tier: Current subscription level
        subscription_status: Current subscription status
        trial_end_date: End date for trial period
        stripe_customer_id: Stripe customer identifier
        stripe_subscription_id: Stripe subscription identifier
        api_requests_count: Total API requests made
        ml_inferences_count: Total ML inferences performed
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """

    id: uuid.UUID = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email: str = models.EmailField(unique=True)
    api_key: str = models.CharField(max_length=128, unique=True, blank=True)

    # Subscription
    subscription_tier: str = models.CharField(
        max_length=20,
        choices=SubscriptionTier.choices,
        default=SubscriptionTier.FREE,
    )
    subscription_status: str = models.CharField(
        max_length=20,
        choices=SubscriptionStatus.choices,
        default=SubscriptionStatus.TRIAL,
    )
    trial_end_date: datetime | None = models.DateTimeField(null=True, blank=True)

    # Billing
    stripe_customer_id: str = models.CharField(max_length=255, blank=True, null=True)
    stripe_subscription_id: str = models.CharField(max_length=255, blank=True, null=True)

    # Usage tracking
    api_requests_count: int = models.IntegerField(default=0)
    ml_inferences_count: int = models.IntegerField(default=0)

    # Timestamps
    created_at: datetime = models.DateTimeField(auto_now_add=True)
    updated_at: datetime = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    if TYPE_CHECKING:
        # Type hints for reverse relations
        payments: RelatedManager
        subscription_detail: RelatedManager

    class Meta:
        db_table = "users"
        verbose_name = "User"
        verbose_name_plural = "Users"
        ordering: ClassVar[list[str]] = ["-created_at"]
        indexes: ClassVar[list] = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["subscription_tier", "subscription_status"]),
        ]

    def __str__(self) -> str:
        return self.email

    def save(self, *args: object, **kwargs: object) -> None:
        """Save user with atomic API key generation.

        Generates unique API key if not present, with retry logic
        to handle potential race conditions.
        """
        if not self.api_key:
            self.api_key = self._generate_api_key()

        try:
            with transaction.atomic():
                super().save(*args, **kwargs)
        except IntegrityError as e:
            # Handle race condition: regenerate key and retry
            if "api_key" in str(e):
                self.api_key = self._generate_api_key()
                super().save(*args, **kwargs)
            else:
                raise

    @staticmethod
    def _generate_api_key() -> str:
        """Generate unique API key."""
        return f"hyd_{secrets.token_urlsafe(32)}"

    @property
    def tier_enum(self) -> SubscriptionTier:
        """Get subscription tier as enum."""
        return SubscriptionTier(self.subscription_tier)

    @property
    def status_enum(self) -> SubscriptionStatus:
        """Get subscription status as enum."""
        return SubscriptionStatus(self.subscription_status)

    @property
    def has_active_subscription(self) -> bool:
        """Check if user has active subscription."""
        return self.status_enum.is_active

    @property
    def api_limit_reached(self) -> bool:
        """Check if API limit is reached for current tier."""
        limit = self.tier_enum.api_limit
        return limit is not None and self.api_requests_count >= limit

    def check_trial_expiration(self) -> None:
        """Check and update trial expiration status.

        Uses match/case for status transition logic.
        """
        match self.subscription_status:
            case SubscriptionStatus.TRIAL if (
                self.trial_end_date and self.trial_end_date < timezone.now()
            ):
                self.subscription_status = SubscriptionStatus.EXPIRED
                self.save(update_fields=["subscription_status", "updated_at"])
            case _:
                pass
