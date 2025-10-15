from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.management import call_command
from django.shortcuts import get_object_or_404

from .models import HydraulicSystem, SensorData, DiagnosticReport
from .serializers import (
    HydraulicSystemListSerializer,
    HydraulicSystemDetailSerializer,
    HydraulicSystemCreateSerializer,
    SensorDataSerializer,
    DiagnosticReportSerializer
)


class HydraulicSystemViewSet(viewsets.ModelViewSet):
    """API для гидравлических систем"""
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return HydraulicSystem.objects.filter(owner=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return HydraulicSystemListSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return HydraulicSystemCreateSerializer
        else:
            return HydraulicSystemDetailSerializer
    
    @action(detail=False, methods=['post'])
    def generate_test_data(self, request):
        """Генерация тестовых данных"""
        try:
            call_command('generate_test_data', systems=5, sensors=50)
            return Response({'message': 'Тестовые данные созданы'}, 
                        status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': f'Ошибка: {str(e)}'}, 
                        status=status.HTTP_400_BAD_REQUEST)
    
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
    permission_classes = []  # Временно убираем аутентификацию для тестирования
    
    def get_queryset(self):
        return SensorData.objects.filter(system__owner=self.request.user)


class DiagnosticReportViewSet(viewsets.ModelViewSet):
    """API для отчетов диагностики с полной поддержкой CRUD"""
    serializer_class = DiagnosticReportSerializer
    permission_classes = []  # Временно убираем аутентификацию для тестирования
    
    def get_queryset(self):
        return DiagnosticReport.objects.filter(system__owner=self.request.user)
