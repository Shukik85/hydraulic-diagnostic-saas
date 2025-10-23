"""Модуль проекта с автогенерированным докстрингом."""

from django.contrib import admin

from .models import (
    Document,
    RagQueryLog,
    RagSystem,
)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "language",
        "format",
        "rag_system",
        "created_at",
        "updated_at",
    ]
    list_filter = ["language", "format", "rag_system", "created_at"]
    search_fields = ["title", "content", "rag_system__name"]
    ordering = ["-created_at"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("rag_system")


@admin.register(RagSystem)
class RagSystemAdmin(admin.ModelAdmin):
    list_display = ["name", "description", "created_at", "updated_at"]
    search_fields = ["name", "description"]
    ordering = ["-created_at"]


@admin.register(RagQueryLog)
class RagQueryLogAdmin(admin.ModelAdmin):
    list_display = ["system", "query_text_preview", "timestamp"]
    search_fields = ["system__name", "query_text"]
    ordering = ["-timestamp"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("system", "document")

    @admin.display(description="Запрос")
    def query_text_preview(self, obj):
        return (
            (obj.query_text[:100] + "...")
            if len(obj.query_text) > 100
            else obj.query_text
        )
