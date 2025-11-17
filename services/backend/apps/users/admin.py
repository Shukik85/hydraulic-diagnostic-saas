"""User admin interface with Django Unfold."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import SafeString
from unfold.admin import ModelAdmin
from unfold.decorators import display

from apps.core.enums import SubscriptionTier

from .models import User

if TYPE_CHECKING:
    pass


@admin.register(User)
class UserAdmin(ModelAdmin):
    """Admin interface for User model with Unfold theme."""

    list_display: ClassVar = [
        "email",
        "subscription_badge",
        "status_badge",
        "api_requests_count",
        "created_at",
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
                "description": "Управление подпиской пользователя и пробным периодом",
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

    @display(description="Tier", label=True)
    def subscription_badge(self, obj: User) -> SafeString:
        """Display subscription tier as badge."""
        tier = obj.tier_enum
        badge_classes = {
            SubscriptionTier.FREE: "secondary",
            SubscriptionTier.PRO: "info",
            SubscriptionTier.ENTERPRISE: "success",
        }
        badge_class = badge_classes.get(tier, "secondary")
        
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            badge_class,
            tier.display_name,
        )

    @display(description="Status", label=True)
    def status_badge(self, obj: User) -> SafeString:
        """Display status badge."""
        if obj.is_active:
            return format_html('<span class="badge bg-success">Активен</span>')
        return format_html('<span class="badge bg-danger">Неактивен</span>')
