"""
Django Admin configuration for quarantine management.

Enterprise data quality review workflow.
"""

from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Count

from diagnostics.models_quarantine import QuarantinedReading
from diagnostics.models_ingestion import IngestionJob


@admin.register(QuarantinedReading)
class QuarantinedReadingAdmin(admin.ModelAdmin):
    """
    Admin interface for reviewing quarantined sensor readings.
    
    Features:
    - Filter by job, reason, review status
    - Bulk actions for approve/reject
    - Inline job details
    - Full audit trail
    """
    
    list_display = [
        'id_short',
        'sensor_id_short',
        'timestamp',
        'value_with_unit',
        'quality_badge',
        'reason_badge',
        'review_status_badge',
        'quarantined_at',
    ]
    
    list_filter = [
        'reason',
        'review_status',
        'unit',
        ('quarantined_at', admin.DateFieldListFilter),
        ('reviewed_at', admin.DateFieldListFilter),
    ]
    
    search_fields = [
        'id',
        'job_id',
        'sensor_id',
        'reason_details',
    ]
    
    readonly_fields = [
        'id',
        'job_id',
        'sensor_id',
        'timestamp',
        'value',
        'unit',
        'quality',
        'system_id',
        'reason',
        'reason_details',
        'quarantined_at',
    ]
    
    fieldsets = (
        ('Sensor Reading', {
            'fields': (
                'sensor_id',
                'timestamp',
                'value',
                'unit',
                'quality',
            )
        }),
        ('Quarantine Info', {
            'fields': (
                'job_id',
                'system_id',
                'reason',
                'reason_details',
                'quarantined_at',
            )
        }),
        ('Review', {
            'fields': (
                'review_status',
                'reviewed_by',
                'reviewed_at',
                'review_notes',
            )
        }),
    )
    
    actions = [
        'approve_and_retry',
        'reject_readings',
        'mark_as_fixed',
    ]
    
    date_hierarchy = 'quarantined_at'
    ordering = ['-quarantined_at']
    
    def id_short(self, obj):
        """Display shortened ID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'
    
    def sensor_id_short(self, obj):
        """Display shortened sensor ID."""
        return str(obj.sensor_id)[:8]
    sensor_id_short.short_description = 'Sensor'
    
    def value_with_unit(self, obj):
        """Display value with unit."""
        return f"{obj.value:.2f} {obj.unit}"
    value_with_unit.short_description = 'Value'
    
    def quality_badge(self, obj):
        """Display quality as colored badge."""
        if obj.quality >= 90:
            color = 'green'
        elif obj.quality >= 70:
            color = 'orange'
        else:
            color = 'red'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            f"{obj.quality}%"
        )
    quality_badge.short_description = 'Quality'
    
    def reason_badge(self, obj):
        """Display reason as colored badge."""
        colors = {
            'out_of_range': 'red',
            'invalid_timestamp': 'orange',
            'duplicate': 'blue',
            'parse_error': 'purple',
            'system_not_found': 'darkred',
            'invalid_unit': 'brown',
        }
        color = colors.get(obj.reason, 'gray')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_reason_display()
        )
    reason_badge.short_description = 'Reason'
    
    def review_status_badge(self, obj):
        """Display review status as colored badge."""
        colors = {
            'pending': 'orange',
            'approved': 'green',
            'rejected': 'red',
            'fixed': 'blue',
        }
        color = colors.get(obj.review_status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px;">{}</span>',
            color,
            obj.get_review_status_display()
        )
    review_status_badge.short_description = 'Review Status'
    
    @admin.action(description='Approve and retry selected readings')
    def approve_and_retry(self, request, queryset):
        """Approve readings for retry."""
        updated = queryset.filter(review_status='pending').update(
            review_status='approved',
            reviewed_by=request.user,
            reviewed_at=admin.models.timezone.now(),
        )
        self.message_user(
            request,
            f"{updated} readings approved for retry.",
            level=admin.messages.SUCCESS,
        )
    
    @admin.action(description='Reject selected readings')
    def reject_readings(self, request, queryset):
        """Reject readings (discard)."""
        updated = queryset.filter(review_status='pending').update(
            review_status='rejected',
            reviewed_by=request.user,
            reviewed_at=admin.models.timezone.now(),
        )
        self.message_user(
            request,
            f"{updated} readings rejected.",
            level=admin.messages.WARNING,
        )
    
    @admin.action(description='Mark as fixed and reprocessed')
    def mark_as_fixed(self, request, queryset):
        """Mark readings as fixed."""
        updated = queryset.update(
            review_status='fixed',
            reviewed_by=request.user,
            reviewed_at=admin.models.timezone.now(),
        )
        self.message_user(
            request,
            f"{updated} readings marked as fixed.",
            level=admin.messages.SUCCESS,
        )


@admin.register(IngestionJob)
class IngestionJobAdmin(admin.ModelAdmin):
    """
    Admin interface for ingestion job monitoring.
    
    Features:
    - Real-time job status tracking
    - Performance metrics
    - Error investigation
    """
    
    list_display = [
        'id_short',
        'status_badge',
        'system_id_short',
        'stats_summary',
        'success_rate_badge',
        'processing_time_badge',
        'created_at',
    ]
    
    list_filter = [
        'status',
        ('created_at', admin.DateFieldListFilter),
        ('completed_at', admin.DateFieldListFilter),
    ]
    
    search_fields = [
        'id',
        'system_id',
        'celery_task_id',
        'error_message',
    ]
    
    readonly_fields = [
        'id',
        'celery_task_id',
        'created_at',
        'started_at',
        'completed_at',
        'processing_time_ms',
        'success_rate_display',
    ]
    
    fieldsets = (
        ('Job Info', {
            'fields': (
                'id',
                'status',
                'system_id',
                'created_by',
            )
        }),
        ('Statistics', {
            'fields': (
                'total_readings',
                'inserted_readings',
                'quarantined_readings',
                'success_rate_display',
            )
        }),
        ('Timing', {
            'fields': (
                'created_at',
                'started_at',
                'completed_at',
                'processing_time_ms',
            )
        }),
        ('Celery', {
            'fields': (
                'celery_task_id',
            )
        }),
        ('Errors', {
            'fields': (
                'error_message',
            ),
            'classes': ('collapse',),
        }),
    )
    
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    def id_short(self, obj):
        """Display shortened ID."""
        return str(obj.id)[:8]
    id_short.short_description = 'ID'
    
    def system_id_short(self, obj):
        """Display shortened system ID."""
        return str(obj.system_id)[:8]
    system_id_short.short_description = 'System'
    
    def status_badge(self, obj):
        """Display status as colored badge."""
        colors = {
            'queued': 'blue',
            'processing': 'orange',
            'completed': 'green',
            'failed': 'red',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            obj.status.upper()
        )
    status_badge.short_description = 'Status'
    
    def stats_summary(self, obj):
        """Display statistics summary."""
        return format_html(
            '<span style="color: green;">✓ {}</span> / '
            '<span style="color: red;">✗ {}</span> / '
            '<span style="color: gray;">Σ {}</span>',
            obj.inserted_readings,
            obj.quarantined_readings,
            obj.total_readings,
        )
    stats_summary.short_description = 'Stats (OK/Quarantine/Total)'
    
    def success_rate_badge(self, obj):
        """Display success rate as colored badge."""
        rate = obj.success_rate
        if rate >= 95:
            color = 'green'
        elif rate >= 80:
            color = 'orange'
        else:
            color = 'red'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
            color,
            rate
        )
    success_rate_badge.short_description = 'Success Rate'
    
    def processing_time_badge(self, obj):
        """Display processing time."""
        if not obj.processing_time_ms:
            return '-'
        
        time_ms = obj.processing_time_ms
        if time_ms < 1000:
            color = 'green'
            text = f"{time_ms}ms"
        elif time_ms < 5000:
            color = 'orange'
            text = f"{time_ms / 1000:.1f}s"
        else:
            color = 'red'
            text = f"{time_ms / 1000:.1f}s"
        
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            text
        )
    processing_time_badge.short_description = 'Processing Time'
    
    def success_rate_display(self, obj):
        """Display detailed success rate."""
        return f"{obj.success_rate:.2f}%"
    success_rate_display.short_description = 'Success Rate'


__all__ = [
    'QuarantinedReadingAdmin',
    'IngestionJobAdmin',
]
