"""Subscription admin interface with type-safe attributes."""

from __future__ import annotations

from typing import ClassVar

from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import Payment, Subscription


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    """Admin interface for subscriptions."""

    list_display: ClassVar[list[str]] = [
        "user_email",
        "tier_badge",
        "status_badge",
        "current_period_end",
        "actions_column",
    ]
    list_filter: ClassVar[list[str]] = ["tier", "status"]
    search_fields: ClassVar[list[str]] = ["user__email", "stripe_subscription_id"]
    readonly_fields: ClassVar[list[str]] = ["stripe_subscription_id", "created_at", "updated_at"]

    def user_email(self, obj: Subscription) -> str:
        """Get user email."""
        return obj.user.email

    user_email.short_description = "User"  # type: ignore[attr-defined]

    def tier_badge(self, obj: Subscription) -> SafeString:
        """Display tier as colored badge."""
        colors = {"free": "gray", "pro": "blue", "enterprise": "green"}
        color = colors.get(obj.tier, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_tier_display(),
        )

    tier_badge.short_description = "Tier"  # type: ignore[attr-defined]

    def status_badge(self, obj: Subscription) -> SafeString:
        """Display status as colored badge."""
        colors = {"active": "green", "trial": "orange", "past_due": "red", "cancelled": "red"}
        color = colors.get(obj.status, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def actions_column(self, obj: Subscription) -> SafeString:
        """Display action buttons."""
        return format_html(
            '<a class="button" href="/admin/subscriptions/subscription/{}/change/">Edit</a>', obj.pk
        )

    actions_column.short_description = "Actions"  # type: ignore[attr-defined]


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    """Admin interface for payments."""

    list_display: ClassVar[list[str]] = [
        "user_email",
        "amount_formatted",
        "status_badge",
        "created_at",
        "invoice_link",
    ]
    list_filter: ClassVar[list[str]] = ["status", "currency", "created_at"]
    search_fields: ClassVar[list[str]] = ["user__email", "stripe_payment_intent_id"]
    readonly_fields: ClassVar[list[str]] = [
        "stripe_payment_intent_id",
        "stripe_invoice_id",
        "created_at",
    ]

    def user_email(self, obj: Payment) -> str:
        """Get user email."""
        return obj.user.email

    user_email.short_description = "User"  # type: ignore[attr-defined]

    def amount_formatted(self, obj: Payment) -> str:
        """Format amount with currency."""
        return f"${obj.amount} {obj.currency}"

    amount_formatted.short_description = "Amount"  # type: ignore[attr-defined]

    def status_badge(self, obj: Payment) -> SafeString:
        """Display status as colored badge."""
        colors = {"succeeded": "green", "pending": "orange", "failed": "red", "refunded": "gray"}
        color = colors.get(obj.status, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def invoice_link(self, obj: Payment) -> SafeString | str:
        """Display invoice link if available."""
        if obj.invoice_url:
            return format_html('<a href="{}" target="_blank">View Invoice</a>', obj.invoice_url)
        return "-"

    invoice_link.short_description = "Invoice"  # type: ignore[attr-defined]
