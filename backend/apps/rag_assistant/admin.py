"""Production RAG Assistant Django Admin with proper FK relations."""

from django.contrib import admin

from .models import Document, RagQueryLog, RagSystem


class DocumentInline(admin.TabularInline):
    """Инлайн для документов в RAG-системе."""
    
    model = Document
    extra = 0
    fields = ("title", "language", "format", "created_at")
    readonly_fields = ("created_at",)
    can_delete = True
    show_change_link = True


@admin.register(RagSystem)
class RagSystemAdmin(admin.ModelAdmin):
    """Продовый админ для RAG-систем."""
    
    list_display = ("name", "description", "model_name", "index_type", "documents_count", "created_at")
    list_filter = ("index_type", ("created_at", admin.DateFieldListFilter))
    search_fields = ("name", "description", "model_name")
    ordering = ("-created_at",)
    inlines = (DocumentInline,)
    
    @admin.display(description="Кол-во документов")
    def documents_count(self, obj):
        """Подсчёт документов."""
        return obj.documents.count()


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Продовый админ для документов."""
    
    list_display = (
        "title",
        "rag_system",
        "language",
        "format",
        "content_preview",
        "created_at",
    )
    list_filter = (
        "language",
        "format",
        "rag_system",
        ("created_at", admin.DateFieldListFilter),
    )
    search_fields = ("title", "content", "rag_system__name")
    ordering = ("-created_at",)
    date_hierarchy = "created_at"
    list_per_page = 25
    
    def get_queryset(self, request):
        """Оптимизация запросов."""
        return super().get_queryset(request).select_related("rag_system")
    
    @admin.display(description="Превью")
    def content_preview(self, obj):
        """Короткое превью содержимого."""
        return (
            obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
        )


@admin.register(RagQueryLog)
class RagQueryLogAdmin(admin.ModelAdmin):
    """Продовый админ для логов запросов."""
    
    list_display = ("system", "query_preview", "document", "timestamp")
    list_filter = (
        "system",
        ("timestamp", admin.DateFieldListFilter),
    )
    search_fields = ("system__name", "query_text", "response_text")
    ordering = ("-timestamp",)
    date_hierarchy = "timestamp"
    list_per_page = 50
    readonly_fields = ("timestamp",)
    
    def get_queryset(self, request):
        """Оптимизация запросов."""
        return super().get_queryset(request).select_related("system", "document")

    @admin.display(description="Запрос")
    def query_preview(self, obj):
        """Короткое превью запроса."""
        return (
            obj.query_text[:100] + "..." if len(obj.query_text) > 100 else obj.query_text
        )
