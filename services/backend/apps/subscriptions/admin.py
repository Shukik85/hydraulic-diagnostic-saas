"""
Subscription admin interface
"""
from django.contrib import admin
from django.utils.html import format_html

from .models import Payment, Subscription


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ['user_email', 'tier_badge', 'status_badge', 'current_period_end', 'actions_column']
    list_filter = ['tier', 'status']
    search_fields = ['user__email', 'stripe_subscription_id']
    readonly_fields = ['stripe_subscription_id', 'created_at', 'updated_at']

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User'

    def tier_badge(self, obj):
        colors = {'free': 'gray', 'pro': 'blue', 'enterprise': 'green'}
        color = colors.get(obj.tier, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_tier_display()
        )
    tier_badge.short_description = 'Tier'

    def status_badge(self, obj):
        colors = {'active': 'green', 'trial': 'orange', 'past_due': 'red', 'cancelled': 'red'}
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def actions_column(self, obj):
        return format_html(
            '<a class="button" href="/admin/subscriptions/subscription/{}/change/">Edit</a>',
            obj.pk
        )
    actions_column.short_description = 'Actions'


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['user_email', 'amount_formatted', 'status_badge', 'created_at', 'invoice_link']
    list_filter = ['status', 'currency', 'created_at']
    search_fields = ['user__email', 'stripe_payment_intent_id']
    readonly_fields = ['stripe_payment_intent_id', 'stripe_invoice_id', 'created_at']

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User'

    def amount_formatted(self, obj):
        return f"${obj.amount} {obj.currency}"
    amount_formatted.short_description = 'Amount'

    def status_badge(self, obj):
        colors = {'succeeded': 'green', 'pending': 'orange', 'failed': 'red', 'refunded': 'gray'}
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def invoice_link(self, obj):
        if obj.invoice_url:
            return format_html('<a href="{}" target="_blank">View Invoice</a>', obj.invoice_url)
        return '-'
    invoice_link.short_description = 'Invoice'
