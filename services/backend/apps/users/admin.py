"""User admin interface with type safety and Friendly UX."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.safestring import SafeString

from apps.core.enums import SubscriptionTier

from .models import User

if TYPE_CHECKING:
    pass


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Admin interface for User model with Friendly UX."""

    list_display: ClassVar = [
        "email",
        "subscription_badge",
        "status_badge",
        "api_requests_count",
        "created_at",
        "actions_column",
    ]
    list_filter: ClassVar = ["subscription_tier", "subscription_status", "is_active", "is_staff"]
    search_fields: ClassVar = ["email", "first_name", "last_name", "api_key"]
    ordering: ClassVar = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Personal Info", {"fields": ("first_name", "last_name")}),
        (
            "Subscription",
            {
                "fields": ("subscription_tier", "subscription_status", "trial_end_date"),
                "description": "<span class='Help'>"
                "<svg class='FormHelperIcon' style='width: 14px; height: 14px; stroke: var(--color-primary); fill: none;'>"
                "<use href='" + static('admin/icons/icons-sprite.svg') + "#icon-info'></use></svg> "
                "Управление подпиской пользователя и пробным периодом"
                "</span>",
            },
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

    readonly_fields: ClassVar = ["created_at", "updated_at", "api_key"]

    def subscription_badge(self, obj: User) -> SafeString:
        """Display subscription tier as FriendlyUX badge with icon."""
        tier = obj.tier_enum
        badge_classes = {
            SubscriptionTier.FREE: "BadgeMuted",
            SubscriptionTier.PRO: "BadgeInfo",
            SubscriptionTier.ENTERPRISE: "BadgeSuccess",
        }
        badge_class = badge_classes.get(tier, "BadgeMuted")

        icon_name = {
            SubscriptionTier.FREE: "icon-users",
            SubscriptionTier.PRO: "icon-star",
            SubscriptionTier.ENTERPRISE: "icon-crown",
        }.get(tier, "icon-users")

        return format_html(
            '<span class="Badge {}">' 
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">' 
            '<use href="{}#{}"></use></svg> '
            "{}"
            "</span>",
            badge_class,
            static("admin/icons/icons-sprite.svg"),
            icon_name,
            tier.display_name,
        )

    subscription_badge.short_description = "Tier"  # type: ignore[attr-defined]

    def status_badge(self, obj: User) -> SafeString:
        """Display status with FriendlyUX badge and icon."""
        if obj.is_active:
            return format_html(
                '<span class="Badge BadgeSuccess">' 
                '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">' 
                '<use href="{}#icon-check"></use></svg> '
                "Активен"
                "</span>",
                static("admin/icons/icons-sprite.svg"),
            )
        return format_html(
            '<span class="Badge BadgeError">' 
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle;">' 
            '<use href="{}#icon-x"></use></svg> '
            "Неактивен"
            "</span>",
            static("admin/icons/icons-sprite.svg"),
        )

    status_badge.short_description = "Status"  # type: ignore[attr-defined]

    def actions_column(self, obj: User) -> SafeString:
        """Display action buttons with FriendlyUX styling."""
        return format_html(
            '<a class="Btn BtnSecondary" style="padding: 6px 12px; font-size: 12px;" href="/admin/users/user/{}/password/">'
            '<svg style="width: 14px; height: 14px; stroke: currentColor; fill: none; vertical-align: middle; margin-right: 4px;">'
            '<use href="{}#icon-key"></use></svg>'
            "Reset Password"
            "</a>",
            obj.pk,
            static("admin/icons/icons-sprite.svg"),
        )

    actions_column.short_description = "Actions"  # type: ignore[attr-defined]
