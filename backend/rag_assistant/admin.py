"""Админка Django для приложения RAG Assistant с правильными отношениями ForeignKey."""

from django.contrib import admin

from .models import Document, RagQueryLog, RagSystem


class DocumentInline(admin.TabularInline):
    """Инлайн для документов в админке RAG-системы."""

    model = Document
    extra = 0
    fields = ("title", "language", "format", "created_at")
    readonly_fields = ("created_at",)
    can_delete = True
    show_change_link = True


@admin.register(RagSystem)
class RagSystemAdmin(admin.ModelAdmin):
    """Админка для RAG-систем с оптимизациями для продакшена."""

    list_display = (
        "name",
        "description",
        "model_name",
        "index_type",
        "documents_count",
        "created_at",
    )
    list_filter = ("index_type", ("created_at", admin.DateFieldListFilter))
    search_fields = ("name", "description", "model_name")
    ordering = ("-created_at",)
    inlines = (DocumentInline,)

    @admin.display(description="Кол-во документов")
    def documents_count(self, obj: RagSystem) -> int:
        """Подсчитывает количество документов в системе.

        Args:
            obj: Экземпляр RAG системы

        Returns:
            Количество документов
        """
        return obj.documents.count()


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Админка для документов с оптимизациями для продакшена."""

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
        """Возвращает оптимизированный queryset с select_related.

        Args:
            request: HTTP запрос

        Returns:
            Оптимизированный QuerySet
        """
        return super().get_queryset(request).select_related("rag_system")

    @admin.display(description="Превью")
    def content_preview(self, obj: Document) -> str:
        """Возвращает короткое превью содержимого документа.

        Args:
            obj: Экземпляр документа

        Returns:
            Превью содержимого (первые 100 символов)
        """
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content


@admin.register(RagQueryLog)
class RagQueryLogAdmin(admin.ModelAdmin):
    """Админка для логов запросов с оптимизациями для продакшена."""

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
        """Возвращает оптимизированный queryset с select_related.

        Args:
            request: HTTP запрос

        Returns:
            Оптимизированный QuerySet
        """
        return super().get_queryset(request).select_related("system", "document")

    @admin.display(description="Запрос")
    def query_preview(self, obj: RagQueryLog) -> str:
        """Возвращает короткое превью запроса.

        Args:
            obj: Экземпляр лога запроса

        Returns:
            Превью запроса (первые 100 символов)
        """
        return (
            obj.query_text[:100] + "..."
            if len(obj.query_text) > 100
            else obj.query_text
        )
