"""DRF ViewSets для приложения RAG Assistant с оптимизациями для продакшена."""

import logging
import time
from typing import Any

from django.db.models import Count
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, RagQueryLog, RagSystem
from .serializers import DocumentSerializer, RagQueryLogSerializer, RagSystemSerializer

# Настройка логирования медленных запросов
logger = logging.getLogger(__name__)


class RagSystemViewSet(viewsets.ModelViewSet):
    """ViewSet для RAG-систем с оптимизациями для продакшена."""

    serializer_class = RagSystemSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["name", "index_type"]
    search_fields = ["name", "description"]
    ordering_fields = ["created_at", "name", "updated_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Возвращает оптимизированный queryset с агрегациями.

        Returns:
            QuerySet с аннотациями количества документов и логов
        """
        return RagSystem.objects.annotate(
            documents_count=Count("documents", distinct=True),
            logs_count=Count("logs", distinct=True),
        ).prefetch_related("documents", "logs")

    @action(detail=True, methods=["get"])
    def stats(self, request: Any, pk: int | None = None) -> Response:
        """Возвращает статистику по RAG системе.

        Args:
            request: HTTP запрос
            pk: ID системы

        Returns:
            Ответ со статистикой системы
        """
        system = self.get_object()

        stats = {
            "total_documents": system.documents.count(),
            "total_logs": system.logs.count(),
            "documents_by_language": dict(
                system.documents.values("language")
                .annotate(count=Count("id"))
                .values_list("language", "count")
            ),
            "documents_by_format": dict(
                system.documents.values("format")
                .annotate(count=Count("id"))
                .values_list("format", "count")
            ),
        }

        return Response(stats)


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet для документов с валидацией для продакшена."""

    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["rag_system", "language", "format"]
    search_fields = ["title", "content", "rag_system__name"]
    ordering_fields = ["created_at", "title", "updated_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Возвращает оптимизированный queryset с eager loading.

        Returns:
            QuerySet с select_related для rag_system
        """
        return Document.objects.select_related("rag_system").all()

    def perform_create(self, serializer):
        """Выполняет кастомную логику при создании документа.

        Args:
            serializer: Сериализатор документа
        """
        document = serializer.save()
        logger.info(
            f"Document created: {document.title} for system {document.rag_system.name}"
        )


class RagQueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ReadOnly ViewSet для логов запросов с оптимизациями."""

    serializer_class = RagQueryLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["system", "document"]
    search_fields = ["query_text", "response_text", "system__name"]
    ordering_fields = ["timestamp"]
    ordering = ["-timestamp"]

    def get_queryset(self):
        """Возвращает оптимизированный queryset с eager loading.

        Returns:
            QuerySet с select_related для system и document
        """
        return RagQueryLog.objects.select_related("system", "document").all()


class HealthCheckView(APIView):
    """APIView для проверки здоровья RAG системы."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Any) -> Response:
        """Выполняет простую проверку доступности системы.

        Args:
            request: HTTP запрос

        Returns:
            Ответ со статусом здоровья системы
        """
        try:
            # Проверяем базовые модели
            systems_count = RagSystem.objects.count()
            documents_count = Document.objects.count()
            logs_count = RagQueryLog.objects.count()

            return Response(
                {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "stats": {
                        "systems": systems_count,
                        "documents": documents_count,
                        "logs": logs_count,
                    },
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e!s}")
            return Response(
                {"status": "unhealthy", "error": str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
