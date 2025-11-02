"""Production-ready RAG Assistant DRF ViewSets with optimizations."""

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
    """Продовый ViewSet для RAG-систем с оптимизациями."""

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
        """Оптимизированный queryset с агрегациями."""
        return RagSystem.objects.annotate(
            documents_count=Count("documents", distinct=True),
            logs_count=Count("logs", distinct=True),
        ).prefetch_related("documents", "logs")

    @action(detail=True, methods=["get"])
    def stats(self, request: Any, pk: int | None = None) -> Response:
        """Получение статистики по системе."""
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
    """Продовый ViewSet для документов с валидацией."""

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
        """Оптимизированный queryset с eager loading."""
        return Document.objects.select_related("rag_system").all()

    def perform_create(self, serializer):
        """Кастомная логика при создании."""
        document = serializer.save()
        logger.info(
            f"Document created: {document.title} for system {document.rag_system.name}"
        )


class RagQueryLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Продовый ReadOnly ViewSet для логов запросов."""

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
        """Оптимизированный queryset с eager loading."""
        return RagQueryLog.objects.select_related("system", "document").all()


class HealthCheckView(APIView):
    """Проверка здоровья RAG системы."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Any) -> Response:
        """Простая проверка доступности."""
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
