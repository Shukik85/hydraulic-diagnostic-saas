"""
Refactored Subscription and Payment models to use shared enums for all status/tier fields.
Business logic includes type-safe accessors and property methods.
"""

import uuid
from typing import ClassVar

from django.db import models

from apps.core.enums import PaymentStatus, SubscriptionStatus, SubscriptionTier
from apps.users.models import User


class Subscription(models.Model):
    """User subscription details"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="subscription_detail")

    tier = models.CharField(
        max_length=20,
        choices=[(t.value, t.description) for t in SubscriptionTier],
        default=SubscriptionTier.FREE.value,
    )
    status = models.CharField(
        max_length=20,
        choices=[(s.value, s.name.title()) for s in SubscriptionStatus],
        default=SubscriptionStatus.TRIAL.value,
    )

    current_period_start = models.DateTimeField()
    current_period_end = models.DateTimeField()
    stripe_subscription_id = models.CharField(max_length=255, blank=True)
    stripe_price_id = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "subscriptions"
        verbose_name = "Subscription"
        verbose_name_plural = "Subscriptions"

    def __str__(self) -> str:
        return f"{self.user.email} - {self.tier_enum.description}"

    @property
    def tier_enum(self) -> SubscriptionTier:
        return SubscriptionTier(self.tier)

    @property
    def status_enum(self) -> SubscriptionStatus:
        return SubscriptionStatus(self.status)

    @property
    def is_active(self) -> bool:
        return self.status_enum.is_active

    @property
    def is_expired(self) -> bool:
        return self.status_enum == SubscriptionStatus.EXPIRED

    @property
    def api_limit(self) -> int | None:
        return self.tier_enum.api_limit

    def activate(self) -> None:
        self.status = SubscriptionStatus.ACTIVE.value
        self.save(update_fields=["status", "updated_at"])

    def expire(self) -> None:
        self.status = SubscriptionStatus.EXPIRED.value
        self.save(update_fields=["status", "updated_at"])


class Payment(models.Model):
    """Payment history"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="payments")

    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    status = models.CharField(
        max_length=20,
        choices=[(p.value, p.name.title()) for p in PaymentStatus],
        default=PaymentStatus.PENDING.value,
    )

    stripe_payment_intent_id = models.CharField(max_length=255, blank=True)
    stripe_invoice_id = models.CharField(max_length=255, blank=True)
    invoice_url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "payments"
        verbose_name = "Payment"
        verbose_name_plural = "Payments"
        ordering: ClassVar[list[str]] = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user.email} - ${self.amount} - {self.status_enum.name.title()}"

    @property
    def status_enum(self) -> PaymentStatus:
        return PaymentStatus(self.status)

    @property
    def is_succeeded(self) -> bool:
        return self.status_enum == PaymentStatus.SUCCEEDED
