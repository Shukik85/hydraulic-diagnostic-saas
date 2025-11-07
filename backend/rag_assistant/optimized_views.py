"""Оптимизированные DRF ViewSets для приложения RAG Assistant."""

import logging
import time

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count, Prefetch
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_headers
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle

from .models import Document, RagQueryLog, RagSystem
from .rag_service import RagAssistant
from .serializers import DocumentSerializer, RagQueryLogSerializer, RagSystemSerializer
from .tasks import index_documents_batch_async, process_document_async

# Логгер для отслеживания производительности
performance_logger = logging.getLogger("performance")


class AIOperationThrottle(UserRateThrottle):
    """Custom throttle для AI операций."""

    scope = "ai_queries"


class OptimizedDocumentViewSet(viewsets.ModelViewSet):
    """Оптимизированный ViewSet для документов.

    Включает:
    - select_related/prefetch_related для оптимизации запросов
    - кеширование списков на 5 минут
    - разные поля для list и detail вьюх
    - async обработка через Celery
    """

    serializer_class = DocumentSerializer
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["language", "format", "rag_system"]
    search_fields = ["title", "content"]
    ordering_fields = ["created_at", "updated_at", "title"]
    ordering = ["-created_at"]
    throttle_classes = [UserRateThrottle, AnonRateThrottle]

    def get_queryset(self):
        """Возвращает оптимизированные запросы с select_related.

        Returns:
            QuerySet с оптимизированными полями в зависимости от действия
        """
        queryset = Document.objects.select_related("rag_system")

        # Разные поля для разных действий
        if self.action == "list":
            # Для списка возвращаем только необходимые поля
            queryset = queryset.only(
                "id",
                "title",
                "language",
                "format",
                "created_at",
                "updated_at",
                "rag_system__id",
                "rag_system__name",
            )

        # Фильтрация по пользователю (будущая функциональность)
        if hasattr(self.request, "user") and self.request.user.is_authenticated:
            if not self.request.user.is_staff:
                # Обычные пользователи видят только публичные документы
                # queryset = queryset.filter(is_public=True)  # Будущая функция
                pass

        return queryset

    @method_decorator(cache_page(60 * 5))  # 5 минут кеш
    @method_decorator(vary_on_headers("User-Agent", "Accept-Language"))
    def list(self, request, *args, **kwargs):
        """Возвращает список документов с кешированием.

        Args:
            request: HTTP запрос
            *args: Дополнительные аргументы
            **kwargs: Дополнительные ключевые аргументы

        Returns:
            Ответ со списком документов
        """
        return super().list(request, *args, **kwargs)

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        """Создает документ с async обработкой.

        Args:
            request: HTTP запрос
            *args: Дополнительные аргументы
            **kwargs: Дополнительные ключевые аргументы

        Returns:
            Ответ с созданным документом и ID задачи
        """
        start_time = time.time()

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Сохраняем документ
        document = serializer.save()

        # Асинхронная обработка и индексация
        task = process_document_async.delay(document.id)

        # Логирование производительности
        duration = time.time() - start_time
        performance_logger.info(
            "Document created and queued for processing",
            extra={
                "document_id": document.id,
                "task_id": task.id,
                "duration_ms": round(duration * 1000, 2),
                "action": "document_create",
            },
        )

        response_data = serializer.data
        response_data["task_id"] = task.id
        response_data["status"] = "created_queued_for_processing"

        return Response(response_data, status=status.HTTP_201_CREATED)

    @action(
        detail=True,
        methods=["post"],
        throttle_classes=[AIOperationThrottle],
        permission_classes=[permissions.IsAuthenticated],
    )
    def reindex(self, request, pk=None):
        """Выполняет переиндексацию конкретного документа.

        Args:
            request: HTTP запрос
            pk: ID документа

        Returns:
            Ответ с информацией о запущенной задаче
        """
        document = self.get_object()

        # Запускаем асинхронную переиндексацию
        task = process_document_async.delay(document.id, reindex=True)

        return Response(
            {
                "message": "Document reindexing started",
                "document_id": document.id,
                "task_id": task.id,
                "status": "queued",
            }
        )

    @action(
        detail=False,
        methods=["post"],
        throttle_classes=[AIOperationThrottle],
        permission_classes=[permissions.IsAuthenticated],
    )
    def batch_reindex(self, request):
        """Выполняет пакетную переиндексацию документов.

        Args:
            request: HTTP запрос

        Returns:
            Ответ с информацией о запущенной задаче

        Raises:
            Response с ошибкой: Если список документов пуст или слишком большой
        """
        document_ids = request.data.get("document_ids", [])

        if not document_ids:
            return Response(
                {"error": "document_ids list is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(document_ids) > 100:
            return Response(
                {"error": "Maximum 100 documents per batch"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Проверяем существование документов
        existing_docs = Document.objects.filter(id__in=document_ids).values_list(
            "id", flat=True
        )

        if len(existing_docs) != len(document_ids):
            missing_ids = set(document_ids) - set(existing_docs)
            return Response(
                {"error": f"Documents not found: {list(missing_ids)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Запускаем пакетную обработку
        task = index_documents_batch_async.delay(list(existing_docs))

        return Response(
            {
                "message": f"Batch reindexing started for {len(existing_docs)} documents",
                "document_count": len(existing_docs),
                "task_id": task.id,
                "status": "queued",
            }
        )


class OptimizedRagSystemViewSet(viewsets.ModelViewSet):
    """Оптимизированный ViewSet для RAG систем."""

    queryset = RagSystem.objects.prefetch_related(
        Prefetch(
            "document_set",
            queryset=Document.objects.only("id", "title", "language", "format"),
        )
    ).annotate(document_count=Count("id"))
    serializer_class = RagSystemSerializer
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    search_fields = ["name", "description"]
    ordering_fields = ["created_at", "name"]
    ordering = ["-created_at"]

    @method_decorator(cache_page(60 * 10))  # 10 минут кеш
    def list(self, request, *args, **kwargs):
        """Возвращает список RAG систем с кешированием.

        Args:
            request: HTTP запрос
            *args: Дополнительные аргументы
            **kwargs: Дополнительные ключевые аргументы

        Returns:
            Ответ со списком систем
        """
        return super().list(request, *args, **kwargs)

    @action(
        detail=True,
        methods=["post"],
        throttle_classes=[AIOperationThrottle],
        permission_classes=[permissions.IsAuthenticated],
    )
    def query(self, request, pk=None):
        """Выполняет оптимизированный запрос к RAG системе.

        Args:
            request: HTTP запрос
            pk: ID системы

        Returns:
            Ответ с результатом запроса

        Raises:
            Response с ошибкой: Если текст запроса отсутствует или произошла ошибка
        """
        start_time = time.time()

        system = self.get_object()
        query_text = request.data.get("query")

        if not query_text:
            return Response(
                {"error": "Query text is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Создаем RAG ассистента
            assistant = RagAssistant(system)

            # Выполняем запрос (с кешированием внутри)
            result = assistant.answer(
                query_text,
                user_id=request.user.id if request.user.is_authenticated else None,
            )

            # Метрики производительности
            duration = time.time() - start_time

            performance_logger.info(
                "RAG query processed",
                extra={
                    "system_id": system.id,
                    "query_length": len(query_text),
                    "response_length": len(result),
                    "duration_ms": round(duration * 1000, 2),
                    "user_id": (
                        request.user.id if request.user.is_authenticated else None
                    ),
                    "action": "rag_query",
                },
            )

            # Получаем статистику кеша
            cache_stats = assistant.get_cache_stats()

            return Response(
                {
                    "answer": result,
                    "system": system.name,
                    "query": query_text,
                    "processing_time_ms": round(duration * 1000, 2),
                    "cache_stats": cache_stats,
                    "timestamp": time.time(),
                }
            )

        except ValidationError as e:
            return Response(
                {"error": "Validation error", "details": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            performance_logger.error(
                "RAG query failed",
                extra={
                    "system_id": system.id,
                    "error": str(e),
                    "duration_ms": round((time.time() - start_time) * 1000, 2),
                    "action": "rag_query_error",
                },
            )
            return Response(
                {"error": "Query processing failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class OptimizedRagQueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Оптимизированный ViewSet для логов RAG запросов.

    ReadOnly - только чтение, создание через RAG service.
    """

    serializer_class = RagQueryLogSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["system", "user"]
    ordering_fields = ["timestamp"]
    ordering = ["-timestamp"]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Возвращает оптимизированные запросы для логов.

        Returns:
            QuerySet с оптимизированными полями в зависимости от действия
        """
        queryset = RagQueryLog.objects.select_related("system", "document")

        # Ограничиваем доступ для обычных пользователей
        if not self.request.user.is_staff:
            queryset = queryset.filter(user=self.request.user)

        # Оптимизация полей для list view
        if self.action == "list":
            queryset = queryset.only(
                "id", "query_text", "timestamp", "system__name", "document__title"
            )

        return queryset

    @method_decorator(cache_page(60 * 2))  # 2 минуты кеш для логов
    def list(self, request, *args, **kwargs):
        """Возвращает список логов запросов с кешированием.

        Args:
            request: HTTP запрос
            *args: Дополнительные аргументы
            **kwargs: Дополнительные ключевые аргументы

        Returns:
            Ответ со списком логов
        """
        return super().list(request, *args, **kwargs)
