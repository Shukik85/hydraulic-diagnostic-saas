"""User admin interface with type safety."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from django.utils.safestring import SafeString

from apps.core.enums import SubscriptionTier

from .models import User

if TYPE_CHECKING:
    pass


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Admin interface for User model."""

    list_display = [
        "email",
        "subscription_badge",
        "status_badge",
        "api_requests_count",
        "created_at",
        "actions_column",
    ]
    list_filter = ["subscription_tier", "subscription_status", "is_active", "is_staff"]
    search_fields = ["email", "first_name", "last_name", "api_key"]
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Personal Info", {"fields": ("first_name", "last_name")}),
        (
            "Subscription",
            {"fields": ("subscription_tier", "subscription_status", "trial_end_date")},
        ),
        (
            "Billing",
            {"fields": ("stripe_customer_id", "stripe_subscription_id")},
        ),
        ("API", {"fields": ("api_key",)}),
        ("Usage", {"fields": ("api_requests_count", "ml_inferences_count")}),
        ("Permissions", {"fields": ("is_active", "is_staff", "is_superuser")}),
        (
            "Important dates",
            {"fields": ("last_login", "created_at", "updated_at")},
        ),
    )

    readonly_fields = ["created_at", "updated_at", "api_key"]

    def subscription_badge(self, obj: User) -> SafeString:
        """Display subscription tier as colored badge."""
        tier = obj.tier_enum
        colors = {
            SubscriptionTier.FREE: "gray",
            SubscriptionTier.PRO: "blue",
            SubscriptionTier.ENTERPRISE: "green",
        }
        color = colors.get(tier, "gray")
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            tier.display_name,
        )

    subscription_badge.short_description = "Tier"  # type: ignore[attr-defined]

    def status_badge(self, obj: User) -> SafeString:
        """Display subscription status as colored badge."""
        status = obj.status_enum
        color = status.display_color
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            status.value.title(),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def actions_column(self, obj: User) -> SafeString:
        """Display action buttons."""
        return format_html(
            '<a class="button" href="/admin/users/user/{}/password/">Reset Password</a>',
            obj.pk,
        )

    actions_column.short_description = "Actions"  # type: ignore[attr-defined]
