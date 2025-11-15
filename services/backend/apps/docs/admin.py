"""Admin configuration for documentation system."""

from __future__ import annotations

from django.contrib import admin
from django.db.models import Count, Q
from django.utils.html import format_html

from .models import Document, DocumentCategory, UserProgress


@admin.register(DocumentCategory)
class DocumentCategoryAdmin(admin.ModelAdmin):
    """Admin interface for DocumentCategory."""

    list_display = (
        "icon_display",
        "name",
        "document_count_display",
        "order",
        "is_active",
        "updated_at",
    )
    list_filter = ("is_active", "created_at")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}
    ordering = ("order", "name")
    readonly_fields = ("created_at", "updated_at", "document_count_display")
    fieldsets = (
        (None, {
            "fields": ("name", "slug", "icon", "description")
        }),
        ("Display", {
            "fields": ("order", "is_active")
        }),
        ("Statistics", {
            "fields": ("document_count_display", "created_at", "updated_at"),
            "classes": ("collapse",)
        }),
    )

    def get_queryset(self, request):
        """Optimize queryset with annotations."""
        queryset = super().get_queryset(request)
        return queryset.annotate(
            doc_count=Count("documents", filter=Q(documents__is_published=True))
        )

    @admin.display(description="Icon")
    def icon_display(self, obj: DocumentCategory) -> str:
        """Display icon with styling."""
        if obj.icon:
            return format_html(
                '<span style="font-size: 1.5em;">{}</span>',
                obj.icon
            )
        return "-"

    @admin.display(description="Documents", ordering="doc_count")
    def document_count_display(self, obj: DocumentCategory) -> str:
        """Display document count with badge."""
        count = getattr(obj, "doc_count", obj.document_count)
        color = "#21808D" if count > 0 else "#6B7280"
        return format_html(
            '<span style="background: {}; color: white; padding: 2px 8px; '
            'border-radius: 12px; font-weight: 550;">{}</span>',
            color,
            count
        )


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Admin interface for Document."""

    list_display = (
        "title",
        "category",
        "is_published",
        "is_featured",
        "view_count",
        "author",
        "updated_at",
    )
    list_filter = (
        "is_published",
        "is_featured",
        "category",
        "created_at",
    )
    search_fields = ("title", "summary", "content", "tags")
    prepopulated_fields = {"slug": ("title",)}
    ordering = ("category__order", "order", "title")
    readonly_fields = ("created_at", "updated_at", "view_count", "preview_link")
    autocomplete_fields = ["author"]
    fieldsets = (
        (None, {
            "fields": ("title", "slug", "category", "summary")
        }),
        ("Content", {
            "fields": ("content",),
            "description": "Use Markdown syntax for formatting"
        }),
        ("Metadata", {
            "fields": ("tags", "author")
        }),
        ("Display Options", {
            "fields": ("order", "is_published", "is_featured")
        }),
        ("Statistics", {
            "fields": ("view_count", "created_at", "updated_at", "preview_link"),
            "classes": ("collapse",)
        }),
    )

    @admin.display(description="Preview")
    def preview_link(self, obj: Document) -> str:
        """Display link to preview document."""
        if obj.pk:
            url = obj.get_absolute_url()
            return format_html(
                '<a href="{}" target="_blank" style="color: #21808D; '
                'font-weight: 550;">View Document →</a>',
                url
            )
        return "-"

    def save_model(self, request, obj, form, change):
        """Auto-set author on creation."""
        if not change and not obj.author:
            obj.author = request.user
        super().save_model(request, obj, form, change)


@admin.register(UserProgress)
class UserProgressAdmin(admin.ModelAdmin):
    """Admin interface for UserProgress."""

    list_display = (
        "user",
        "document",
        "completed",
        "last_viewed_at",
    )
    list_filter = (
        "completed",
        "last_viewed_at",
        "document__category",
    )
    search_fields = (
        "user__email",
        "user__first_name",
        "user__last_name",
        "document__title",
    )
    readonly_fields = ("created_at", "last_viewed_at")
    autocomplete_fields = ["user", "document"]
    ordering = ("-last_viewed_at",)
