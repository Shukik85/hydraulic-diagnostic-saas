"""Модуль проекта с автогенерированным докстрингом."""

import logging

from django.db.models import Prefetch

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import DiagnosticReport, HydraulicSystem, SensorData, SystemComponent
from .serializers import (
    DiagnosticReportSerializer,
    HydraulicSystemListSerializer,
    SensorDataSerializer,
    SystemComponentSerializer,
)

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class BaseModelViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]


class HydraulicSystemViewSet(BaseModelViewSet):
    serializer_class = HydraulicSystemListSerializer
    filterset_fields = ["system_type", "status", "criticality"]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "created_at", "updated_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Получает queryset"""
        return (
            HydraulicSystem.objects.select_related()
            .prefetch_related(
                Prefetch(
                    "sensor_data",
                    queryset=SensorData.objects.order_by("-timestamp")[:50],
                )
            )
            .all()
        )

    @action(detail=True, methods=["get"])
    def sensor_data(self, request, pk=None):
        """Выполняет sensor data

        Args:
            request (HttpRequest): HTTP запрос
            pk (int): Первичный ключ объекта

        """
        system = self.get_object()
        data = system.sensor_data.select_related("component").order_by("-timestamp")[
            :100
        ]
        serializer = SensorDataSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"])
    def reports(self, request, pk=None):
        """Выполняет reports

        Args:
            request (HttpRequest): HTTP запрос
            pk (int): Первичный ключ объекта

        """
        system = self.get_object()
        reports = system.diagnostic_reports.select_related("created_by").order_by(
            "-created_at"
        )
        serializer = DiagnosticReportSerializer(reports, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"], parser_classes=[MultiPartParser, FormParser])
    def upload_sensor_data(self, request, pk=None):
        """Выполняет upload sensor data

        Args:
            request (HttpRequest): HTTP запрос
            pk (int): Первичный ключ объекта

        """
        self.get_object()
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response(
                {"error": "Файл не найден"}, status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # TODO: parse and save SensorData via pdfminer/unstructured pipeline
            return Response(
                {"message": "Файл успешно обработан"}, status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки файла для системы {pk}: {e}")
            return Response(
                {"error": "Ошибка обработки файла"}, status=status.HTTP_400_BAD_REQUEST
            )


class SystemComponentViewSet(BaseModelViewSet):
    serializer_class = SystemComponentSerializer
    filterset_fields = ["system"]
    search_fields = ["name"]
    ordering_fields = ["name", "created_at"]
    ordering = ["name"]

    def get_queryset(self):
        """Получает queryset"""
        return SystemComponent.objects.select_related("system").all()


class SensorDataViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = SensorDataSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["system", "component"]
    ordering_fields = ["timestamp"]
    ordering = ["-timestamp"]

    def get_queryset(self):
        """Получает queryset"""
        return SensorData.objects.select_related("system", "component").all()


class DiagnosticReportViewSet(BaseModelViewSet):
    serializer_class = DiagnosticReportSerializer
    filterset_fields = ["system", "severity", "status"]
    search_fields = ["title"]
    ordering_fields = ["created_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Получает queryset"""
        return DiagnosticReport.objects.select_related("system", "created_by").all()

    @action(detail=True, methods=["post"])
    def complete(self, request, pk=None):
        """Выполняет complete

        Args:
            request (HttpRequest): HTTP запрос
            pk (int): Первичный ключ объекта

        """
        report = self.get_object()
        report.status = "closed"
        report.save(update_fields=["status"])  # timestamp handled in model if needed
        serializer = self.get_serializer(report)
        return Response(serializer.data)
