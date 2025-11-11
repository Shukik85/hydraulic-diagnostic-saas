"""
Support admin interface (UPDATED)
"""
from django.contrib import admin
from django.utils.html import format_html
from .models import SupportTicket, SupportAction, DataExportRequest


@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin):
    list_display = ['id_short', 'subject', 'priority_badge', 'status_badge', 'user_id', 'created_at', 'actions_column']
    list_filter = ['status', 'priority', 'created_at']
    search_fields = ['subject', 'message', 'user_id']
    readonly_fields = ['id', 'user_id', 'created_at', 'updated_at']

    fieldsets = (
        ('Ticket Info', {
            'fields': ('id', 'user_id', 'subject', 'message', 'category')
        }),
        ('Priority & Status', {
            'fields': ('priority', 'status', 'assigned_to')
        }),
        ('Response', {
            'fields': ('response',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )

    def id_short(self, obj):
        return str(obj.id)[:8] + "..."
    id_short.short_description = 'ID'

    def priority_badge(self, obj):
        colors = {'low': '#6c757d', 'medium': '#ffc107', 'high': '#dc3545'}
        color = colors.get(obj.priority, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_priority_display()
        )
    priority_badge.short_description = 'Priority'

    def status_badge(self, obj):
        colors = {
            'open': '#dc3545',
            'in_progress': '#0dcaf0',
            'resolved': '#198754',
            'closed': '#6c757d'
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def actions_column(self, obj):
        return format_html(
            '<a class="button" href="/admin/support/supportticket/{}/change/">Respond</a>',
            obj.pk
        )
    actions_column.short_description = 'Actions'


@admin.register(DataExportRequest)
class DataExportRequestAdmin(admin.ModelAdmin):
    list_display = ['id_short', 'user_id', 'status_badge', 'created_at', 'completed_at', 'download_link']
    list_filter = ['status', 'created_at']
    search_fields = ['user_id']
    readonly_fields = ['id', 'user_id', 'created_at', 'completed_at', 'expires_at']

    def id_short(self, obj):
        return str(obj.id)[:8] + "..."
    id_short.short_description = 'ID'

    def status_badge(self, obj):
        colors = {
            'pending': '#ffc107',
            'processing': '#0dcaf0',
            'completed': '#198754',
            'failed': '#dc3545'
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def download_link(self, obj):
        if obj.download_url:
            return format_html('<a href="{}" target="_blank">Download</a>', obj.download_url)
        return '-'
    download_link.short_description = 'Download'


@admin.register(SupportAction)
class SupportActionAdmin(admin.ModelAdmin):
    list_display = ['user_id', 'action_type', 'performed_by', 'created_at']
    list_filter = ['action_type', 'created_at']
    search_fields = ['user_id', 'description']
    readonly_fields = ['id', 'created_at']
