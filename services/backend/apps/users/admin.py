"""
User admin interface
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from .models import User

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['email', 'subscription_badge', 'status_badge', 'api_requests_count', 'created_at', 'actions_column']
    list_filter = ['subscription_tier', 'subscription_status', 'is_active', 'is_staff']
    search_fields = ['email', 'first_name', 'last_name', 'api_key']
    ordering = ['-created_at']

    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {'fields': ('first_name', 'last_name')}),
        ('Subscription', {'fields': ('subscription_tier', 'subscription_status', 'trial_end_date')}),
        ('Billing', {'fields': ('stripe_customer_id', 'stripe_subscription_id')}),
        ('API', {'fields': ('api_key',)}),
        ('Usage', {'fields': ('api_requests_count', 'ml_inferences_count')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser')}),
        ('Important dates', {'fields': ('last_login', 'created_at', 'updated_at')}),
    )

    readonly_fields = ['created_at', 'updated_at', 'api_key']

    def subscription_badge(self, obj):
        colors = {'free': 'gray', 'pro': 'blue', 'enterprise': 'green'}
        color = colors.get(obj.subscription_tier, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_subscription_tier_display()
        )
    subscription_badge.short_description = 'Tier'

    def status_badge(self, obj):
        colors = {'active': 'green', 'trial': 'orange', 'cancelled': 'red', 'expired': 'gray'}
        color = colors.get(obj.subscription_status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_subscription_status_display()
        )
    status_badge.short_description = 'Status'

    def actions_column(self, obj):
        return format_html(
            '<a class="button" href="/admin/users/user/{}/password/">Reset Password</a>',
            obj.pk
        )
    actions_column.short_description = 'Actions'
