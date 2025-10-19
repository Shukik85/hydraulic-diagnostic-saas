import logging

from django.db.models import Prefetch
from django.utils import timezone
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import (
    DiagnosticReport,
    HydraulicSystem,
    MaintenanceSchedule,
    SensorData,
    SystemComponent,
)
from .serializers import (
    DiagnosticReportSerializer,
    HydraulicSystemListSerializer,
    MaintenanceScheduleSerializer,
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
        system = self.get_object()
        data = system.sensor_data.select_related("component").order_by("-timestamp")[
            :100
        ]
        serializer = SensorDataSerializer(data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"])
    def reports(self, request, pk=None):
        system = self.get_object()
        reports = system.diagnostic_reports.select_related("created_by").order_by(
            "-created_at"
        )
        serializer = DiagnosticReportSerializer(reports, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def diagnose(self, request, pk=None):
        system = self.get_object()
        try:
            report = DiagnosticReport.objects.create(
                system=system,
                title=f"Автоматическая диагностика - {timezone.now():%Y-%m-%d %H:%M}",
                severity="info",
                status="pending",
                created_by=request.user,
            )
            # TODO: integrate DiagnosticEngine here
            serializer = DiagnosticReportSerializer(report)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Ошибка диагностики системы {pk}: {e}")
            return Response(
                {"error": "Ошибка при выполнении диагностики"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=True, methods=["post"], parser_classes=[MultiPartParser, FormParser])
    def upload_sensor_data(self, request, pk=None):
        self.get_object()
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response(
                {"error": "Файл не найден"}, status=status.HTTP_400_BAD_REQUEST
            )
        try:
            # TODO: parse and save SensorData via pandas or openpyxl
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
        return SensorData.objects.select_related("system", "component").all()


class DiagnosticReportViewSet(BaseModelViewSet):
    serializer_class = DiagnosticReportSerializer
    filterset_fields = ["system", "severity", "status"]
    search_fields = ["title"]
    ordering_fields = ["created_at", "completed_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        return DiagnosticReport.objects.select_related("system", "created_by").all()

    @action(detail=True, methods=["post"])
    def complete(self, request, pk=None):
        report = self.get_object()
        report.mark_completed()
        serializer = self.get_serializer(report)
        return Response(serializer.data)


class MaintenanceScheduleViewSet(BaseModelViewSet):
    serializer_class = MaintenanceScheduleSerializer
    filterset_fields = ["system"]
    ordering_fields = ["schedule_date"]
    ordering = ["schedule_date"]

    def get_queryset(self):
        return MaintenanceSchedule.objects.select_related("system").all()
