from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg
from django.utils import timezone
from django.shortcuts import get_object_or_404
from datetime import timedelta
import pandas as pd
import logging

from .models import (
    HydraulicSystem, SensorData, SensorType, DiagnosticReport,
    MaintenanceRecord, SystemComponent, Alert
)
from .serializers import (
    HydraulicSystemListSerializer, HydraulicSystemDetailSerializer,
    HydraulicSystemCreateSerializer,
    SensorDataSerializer, SensorDataBulkSerializer, SensorTypeSerializer,
    DiagnosticReportListSerializer, DiagnosticReportDetailSerializer,
    DiagnosticReportSerializer,
    MaintenanceRecordListSerializer, MaintenanceRecordDetailSerializer,
    SystemComponentSerializer, AlertSerializer, SystemStatsSerializer,
    DiagnosticRequestSerializer
)
from .ai_engine import get_ai_engine
from apps.users.models import UserActivity

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class HydraulicSystemViewSet(viewsets.ModelViewSet):
    """ViewSet для гидравлических систем"""
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system_type', 'status', 'criticality']
    search_fields = ['name', 'description', 'location']
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['-created_at']

    def get_queryset(self):
        return HydraulicSystem.objects.filter(owner=self.request.user)

    def get_serializer_class(self):
        if self.action == 'list':
            return HydraulicSystemListSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return HydraulicSystemCreateSerializer
        return HydraulicSystemDetailSerializer

    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_sensor_data(self, request, pk=None):
        """Загрузка данных датчиков из файла"""
        system = self.get_object()
        
        if 'file' not in request.FILES:
            return Response({'error': 'Файл не найден'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Здесь будет логика обработки файла
        return Response({'message': 'Файл загружен успешно'}, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['get'])
    def sensor_data(self, request, pk=None):
        """Получить данные датчиков для системы"""
        system = self.get_object()
        sensor_data = system.sensor_data.all()[:50]  # Последние 50 записей
        serializer = SensorDataSerializer(sensor_data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def reports(self, request, pk=None):
        """Получить отчеты для системы"""
        system = self.get_object()
        reports = system.reports.all()
        serializer = DiagnosticReportSerializer(reports, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def diagnose(self, request, pk=None):
        """Запустить диагностику системы"""
        system = self.get_object()
        # Здесь будет логика диагностики
        # Пока создаем тестовый отчет
        report = DiagnosticReport.objects.create(
            system=system,
            title="Автоматическая диагностика",
            description="Система работает в нормальном режиме",
            severity="info",
            ai_analysis="Все параметры в пределах нормы",
            recommendations="Продолжить мониторинг"
        )
        serializer = DiagnosticReportSerializer(report)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class SensorDataViewSet(viewsets.ReadOnlyModelViewSet):
    """API для данных датчиков (только чтение)"""
    serializer_class = SensorDataSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SensorData.objects.filter(system__owner=self.request.user)


class DiagnosticReportViewSet(viewsets.ModelViewSet):
    """API для отчетов диагностики с полной поддержкой CRUD"""
    serializer_class = DiagnosticReportSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return DiagnosticReport.objects.filter(system__owner=self.request.user)
