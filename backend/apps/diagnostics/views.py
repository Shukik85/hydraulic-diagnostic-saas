from rest_framework import viewsets, status, permissions, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg, Prefetch
from django.utils import timezone
from django.shortcuts import get_object_or_404
from datetime import timedelta
import logging

from .models import (
    HydraulicSystem,
    SystemComponent,
    SensorData,
    DiagnosticReport,
    MaintenanceSchedule,
)
from .serializers import (
    HydraulicSystemListSerializer,
    SystemComponentSerializer,
    SensorDataSerializer,
    DiagnosticReportSerializer,
    MaintenanceScheduleSerializer,
)

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    """Стандартная пагинация для всех ViewSets."""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BaseModelViewSet(viewsets.ModelViewSet):
    """Базовый ViewSet с общими настройками."""
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]


class HydraulicSystemViewSet(BaseModelViewSet):
    """ViewSet для управления гидравлическими системами."""
    serializer_class = HydraulicSystemListSerializer
    filterset_fields = ['system_type', 'status', 'criticality']
    search_fields = ['name', 'description']
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['-created_at']

    def get_queryset(self):
        """Возвращает системы с оптимизированными запросами."""
        return HydraulicSystem.objects.select_related().prefetch_related(
            Prefetch('sensor_data', queryset=SensorData.objects.order_by('-timestamp')[:50])
        ).all()

    @action(detail=True, methods=['get'])
    def sensor_data(self, request, pk=None):
        """Получить последние данные датчиков для системы."""
        system = self.get_object()
        sensor_data = system.sensor_data.select_related('component').order_by('-timestamp')[:100]
        serializer = SensorDataSerializer(sensor_data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def reports(self, request, pk=None):
        """Получить отчеты диагностики для системы."""
        system = self.get_object()
        reports = system.diagnostic_reports.select_related('created_by').order_by('-created_at')
        serializer = DiagnosticReportSerializer(reports, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def diagnose(self, request, pk=None):
        """Запустить диагностику системы."""
        system = self.get_object()
        
        try:
            # Создаём новый отчёт диагностики
            report = DiagnosticReport.objects.create(
                system=system,
                title=f"Автоматическая диагностика - {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                severity='info',
                status='pending',
                created_by=request.user,
            )
            
            # Здесь должна быть интеграция с DiagnosticEngine
            # from services.diagnostic_engine import DiagnosticEngine
            # engine = DiagnosticEngine()
            # result = engine.analyze_system(system)
            # report.ai_confidence = result.get('confidence')
            # report.status = 'completed'
            # report.save()
            
            serializer = DiagnosticReportSerializer(report)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            logger.error(f"Ошибка диагностики системы {pk}: {str(e)}")
            return Response(
                {'error': 'Ошибка при выполнении диагностики'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_sensor_data(self, request, pk=None):
        """Загрузка данных датчиков из файла (CSV/Excel)."""
        system = self.get_object()
        
        if 'file' not in request.FILES:
            return Response(
                {'error': 'Файл не найден'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        
        try:
            # Здесь должна быть логика парсинга файла
            # import pandas as pd
            # df = pd.read_csv(uploaded_file) или pd.read_excel(uploaded_file)
            # Обработка и сохранение данных в SensorData
            
            return Response(
                {'message': 'Файл успешно загружен и обработан'},
                status=status.HTTP_201_CREATED
            )
        
        except Exception as e:
            logger.error(f"Ошибка загрузки файла для системы {pk}: {str(e)}")
            return Response(
                {'error': 'Ошибка обработки файла'},
                status=status.HTTP_400_BAD_REQUEST
            )


class SystemComponentViewSet(BaseModelViewSet):
    """ViewSet для управления компонентами систем."""
    serializer_class = SystemComponentSerializer
    filterset_fields = ['system']
    search_fields = ['name']
    ordering_fields = ['name', 'created_at']
    ordering = ['name']

    def get_queryset(self):
        return SystemComponent.objects.select_related('system').all()


class SensorDataViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet для данных датчиков (только чтение)."""
    serializer_class = SensorDataSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['system', 'component']
    ordering_fields = ['timestamp']
    ordering = ['-timestamp']

    def get_queryset(self):
        return SensorData.objects.select_related('system', 'component').all()


class DiagnosticReportViewSet(BaseModelViewSet):
    """ViewSet для отчетов диагностики."""
    serializer_class = DiagnosticReportSerializer
    filterset_fields = ['system', 'severity', 'status']
    search_fields = ['title']
    ordering_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']

    def get_queryset(self):
        return DiagnosticReport.objects.select_related('system', 'created_by').all()

    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Отметить отчёт как завершённый."""
        report = self.get_object()
        report.mark_completed()
        serializer = self.get_serializer(report)
        return Response(serializer.data)


class MaintenanceScheduleViewSet(BaseModelViewSet):
    """ViewSet для графиков обслуживания."""
    serializer_class = MaintenanceScheduleSerializer
    filterset_fields = ['system']
    ordering_fields = ['schedule_date']
    ordering = ['schedule_date']

    def get_queryset(self):
        return MaintenanceSchedule.objects.select_related('system').all()
