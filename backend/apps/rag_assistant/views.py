"""Модуль проекта с автогенерированным докстрингом."""

# apps/rag_assistant/views.py
import logging
import time

from django.db.models import Avg, Count, Prefetch
from django.db.models.functions import Length

from celery.result import AsyncResult
from rest_framework import filters, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, RagQueryLog, RagSystem
from .rag_service import RagAssistant
from .serializers import DocumentSerializer, RagQueryLogSerializer, RagSystemSerializer
from .tasks import index_documents_batch_async

# Настройка логирования медленных запросов
logger = logging.getLogger(__name__)


class TimingMiddleware:
    """Middleware для логирования медленных запросов"""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        response = self.get_response(request)
        duration = (time.time() - start_time) * 1000  # в миллисекундах

        if duration > 100:  # если запрос дольше 100ms
            logger.warning(f"Slow query: {request.path} took {duration:.2f}ms")

        return response


class DocumentViewSet(viewsets.ModelViewSet):
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["title", "content"]
    ordering_fields = ["created_at", "language"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Оптимизированный queryset с eager loading"""
        queryset = (
            Document.objects.select_related(
                "user"
            )  # если есть ForeignKey на пользователя
            .prefetch_related(
                # prefetch для ManyToMany отношений, если есть
            )
            .annotate(content_length=Length("content"))
        )

        # Фильтрация по системе, если указано
        system_id = self.request.query_params.get("system_id")
        if system_id:
            queryset = queryset.filter(metadata__rag_system=system_id)

        return queryset


class RagSystemViewSet(viewsets.ModelViewSet):
    serializer_class = RagSystemSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Оптимизированный queryset с агрегациями"""
        return RagSystem.objects.annotate(
            document_count=Count("documents", distinct=True),
            query_count=Count("query_logs", distinct=True),
        ).prefetch_related(
            Prefetch(
                "documents",
                queryset=Document.objects.select_related().only(
                    "id", "title", "format", "language", "created_at"
                ),
            ),
            Prefetch(
                "query_logs",
                queryset=RagQueryLog.objects.select_related("document").only(
                    "id", "query_text", "timestamp"
                )[
                    :10
                ],  # Ограничиваем количество для оптимизации
            ),
        )

    @action(detail=True, methods=["post"])
    def index(self, request, pk=None):
        """Асинхронная индексация документов"""
        system = self.get_object()

        # Получение списка документов для индексации
        document_ids = request.data.get("document_ids")

        # Если список не передан — индексируем все документы системы
        if not document_ids:
            document_ids = list(
                Document.objects.filter(rag_system=system).values_list("id", flat=True)
            )

        # Запуск асинхронной пакетной задачи
        task = index_documents_batch_async.delay(document_ids)

        return Response(
            {
                "task_id": task.id,
                "status": "started",
                "message": "Documents indexing started",
            },
            status=status.HTTP_202_ACCEPTED,
        )

    @action(detail=True, methods=["post"])
    def query(self, request, pk=None):
        """Синхронный запрос к RAG системе"""
        system = self.get_object()
        text = request.data.get("query")

        if not text:
            return Response(
                {"error": "Query text is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Измеряем время выполнения запроса
            start_time = time.time()
            assistant = RagAssistant(system)
            answer = assistant.answer(text)
            duration = (time.time() - start_time) * 1000

            # Логируем медленные запросы
            if duration > 100:
                logger.warning(
                    f"Slow RAG query processing: {duration:.2f}ms for query: {text[:50]}..."
                )

            _ = RagQueryLog.objects.create(
                system=system, query_text=text, response_text=answer
            )
            return Response({"answer": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=["get"])
    def stats(self, request, pk=None):
        """Получение статистики по системе"""
        system = self.get_object()

        # Batch loading для агрегаций
        stats = RagQueryLog.objects.filter(system=system).aggregate(
            total_queries=Count("id"),
            avg_response_time=Avg("response_time"),  # если есть такое поле
            total_documents=Count("document", distinct=True),
        )

        # Дополнительные статистики
        recent_queries = (
            RagQueryLog.objects.filter(system=system)
            .select_related("document")
            .order_by("-timestamp")[:10]
        )

        return Response(
            {
                "system_stats": stats,
                "recent_queries": RagQueryLogSerializer(recent_queries, many=True).data,
            }
        )


class RagQueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = RagQueryLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter, filters.SearchFilter]
    ordering_fields = ["timestamp", "system__name"]
    ordering = ["-timestamp"]
    search_fields = ["query_text", "response_text"]

    def get_queryset(self):
        """Оптимизированный queryset с eager loading"""
        return (
            RagQueryLog.objects.select_related("system", "document")
            .annotate(
                query_length=Length("query_text"),
                response_length=Length("response_text"),
            )
            .prefetch_related(
                # Дополнительные prefetch, если есть ManyToMany отношения
            )
        )


class TaskStatusView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, task_id):
        """Получение статуса задачи по ID"""
        task_result = AsyncResult(str(task_id))

        response_data = {
            "task_id": str(task_id),
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
        }

        if task_result.failed():
            response_data["error"] = str(task_result.info)

        return Response(response_data)
