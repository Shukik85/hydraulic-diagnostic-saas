"""Subscription admin interface with Friendly UX."""

from __future__ import annotations

from typing import ClassVar

from django.contrib import admin
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.safestring import SafeString

from .models import Payment, Subscription


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    """Admin interface for subscriptions with Friendly UX."""

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
        """Display tier with FriendlyUX badge and icon."""
        badge_classes = {
            "free": "BadgeMuted",
            "pro": "BadgeInfo",
            "enterprise": "BadgeSuccess",
        }
        badge_class = badge_classes.get(obj.tier, "BadgeMuted")

        icon_name = {
            "free": "icon-users",
            "pro": "icon-star",
            "enterprise": "icon-crown",
        }.get(obj.tier, "icon-users")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_tier_display(),
        )

    tier_badge.short_description = "Tier"  # type: ignore[attr-defined]

    def status_badge(self, obj: Subscription) -> SafeString:
        """Display status with FriendlyUX badge and icon."""
        badge_classes = {
            "active": "BadgeSuccess",
            "trial": "BadgeWarning",
            "past_due": "BadgeError",
            "cancelled": "BadgeError",
        }
        badge_class = badge_classes.get(obj.status, "BadgeMuted")

        icon_name = {
            "active": "icon-check",
            "trial": "icon-clock",
            "past_due": "icon-alert",
            "cancelled": "icon-x",
        }.get(obj.status, "icon-minus")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def actions_column(self, obj: Subscription) -> SafeString:
        """Display action buttons with FriendlyUX styling."""
        return format_html(
            '<a class="Btn BtnSecondary" style="padding: 6px 12px; font-size: 12px;" href="/admin/subscriptions/subscription/{}/change/">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle; margin-right: 4px;">'
            '<use href="{}#icon-edit"></use></svg>'
            "Edit"
            "</a>",
            obj.pk,
            static("admin/icons/icons-sprite.svg"),
        )

    actions_column.short_description = "Actions"  # type: ignore[attr-defined]


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    """Admin interface for payments with Friendly UX."""

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
        """Display status with FriendlyUX badge and icon."""
        badge_classes = {
            "succeeded": "BadgeSuccess",
            "pending": "BadgeWarning",
            "failed": "BadgeError",
            "refunded": "BadgeMuted",
        }
        badge_class = badge_classes.get(obj.status, "BadgeMuted")

        icon_name = {
            "succeeded": "icon-check",
            "pending": "icon-clock",
            "failed": "icon-x",
            "refunded": "icon-refresh",
        }.get(obj.status, "icon-minus")

        return format_html(
            '<span class="Badge {}">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">'
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def invoice_link(self, obj: Payment) -> SafeString | str:
        """Display invoice link with FriendlyUX button if available."""
        if obj.invoice_url:
            return format_html(
                '<a class="Btn BtnSecondary" style="padding: 4px 10px; font-size: 11px;" href="{}" target="_blank">'
                '<svg style="width: 12px; height: 12px; stroke: currentColor; fill: none; vertical-align: middle; margin-right: 4px;">'
                '<use href="{}#icon-external"></use></svg>'
                "View Invoice"
                "</a>",
                obj.invoice_url,
                static("admin/icons/icons-sprite.svg"),
            )
        return "-"

    invoice_link.short_description = "Invoice"  # type: ignore[attr-defined]
