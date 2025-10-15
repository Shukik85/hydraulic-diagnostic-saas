from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg
from django.utils import timezone
from django.core.management import call_command
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
    MaintenanceRecordListSerializer, MaintenanceRecordDetailSerializer,
    SystemComponentSerializer, AlertSerializer, SystemStatsSerializer,
    DiagnosticReportSerializer, DiagnosticRequestSerializer
)
from .ai_engine import get_ai_engine
from apps.users.models import UserActivity
logger = logging.getLogger(__name__)

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

class HydraulicSystemViewSet(viewsets.ModelViewSet):
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
        else:
            return HydraulicSystemDetailSerializer

    def perform_create(self, serializer):
        system = serializer.save(owner=self.request.user)
        UserActivity.objects.create(
            user=self.request.user,
            action='system_created',
            description=f'Создана система: {system.name}',
            metadata={'system_id': str(system.id)}
        )

    def perform_update(self, serializer):
        system = serializer.save()
        UserActivity.objects.create(
            user=self.request.user,
            action='system_updated',
            description=f'Обновлена система: {system.name}',
            metadata={'system_id': str(system.id)}
        )

    def perform_destroy(self, instance):
        system_name = instance.name
        system_id = str(instance.id)
        super().perform_destroy(instance)
        UserActivity.objects.create(
            user=self.request.user,
            action='system_deleted',
            description=f'Удалена система: {system_name}',
            metadata={'system_id': system_id}
        )

    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_sensor_data(self, request, pk=None):
        system = self.get_object()
        if 'file' not in request.FILES:
            return Response({'error': 'Файл не найден'}, status=status.HTTP_400_BAD_REQUEST)
        file = request.FILES['file']
        max_size = 10 * 1024 * 1024
        if file.size > max_size:
            return Response({'error': 'Файл слишком большой (макс 10MB)'}, status=status.HTTP_400_BAD_REQUEST)
        allowed_ext = {'.csv', '.xlsx', '.json'}
        allowed_ct = {'text/csv','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','application/json','text/plain'}
        name = file.name.lower()
        ext = '.' + name.split('.')[-1] if '.' in name else ''
        if ext not in allowed_ext or (file.content_type and file.content_type not in allowed_ct):
            return Response({'error': 'Неподдерживаемый формат файла'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            if ext == '.csv':
                df = pd.read_csv(file)
            elif ext == '.xlsx':
                df = pd.read_excel(file)
            elif ext == '.json':
                df = pd.read_json(file)
            else:
                return Response({'error': 'Неподдерживаемый формат'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': 'Не удалось прочитать файл', 'details': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        required_cols = {'sensor_type', 'value', 'unit', 'timestamp'}
        if not required_cols.issubset({c.lower() for c in df.columns}):
            return Response({'error': 'Отсутствуют обязательные колонки: sensor_type, value, unit, timestamp'}, status=status.HTTP_400_BAD_REQUEST)
        created = 0
        for _, row in df.iterrows():
            try:
                sensor_type_name = row.get('sensor_type') or row.get('SENSOR_TYPE') or row.get('SensorType')
                value = row.get('value') or row.get('VALUE')
                unit = row.get('unit') or row.get('UNIT')
                timestamp = row.get('timestamp') or row.get('TIMESTAMP')
                sensor_type_obj, _ = SensorType.objects.get_or_create(name=str(sensor_type_name).strip())
                SensorData.objects.create(system=system, sensor_type=sensor_type_obj, value=value, unit=unit, timestamp=timestamp)
                created += 1
            except Exception:
                continue
        return Response({'message': 'Файл загружен успешно', 'created': created}, status=status.HTTP_201_CREATED)

class SensorDataViewSet(viewsets.ModelViewSet):
    serializer_class = SensorDataSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['sensor_type', 'is_critical', 'status']
    search_fields = ['unit']
    ordering_fields = ['timestamp', 'value']
    ordering = ['-timestamp']

    def get_queryset(self):
        return SensorData.objects.filter(system__owner=self.request.user)

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        serializer = SensorDataBulkSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = serializer.save()
        return Response({'message': f'Создано {result["created_count"]} записей', 'created_count': result['created_count']}, status=status.HTTP_201_CREATED)

class DiagnosticReportViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'report_type', 'severity', 'status']
    search_fields = ['title', 'description']
    ordering_fields = ['created_at', 'severity']
    ordering = ['-created_at']

    def get_queryset(self):
        return DiagnosticReport.objects.filter(system__owner=self.request.user)

    def get_serializer_class(self):
        if self.action == 'list':
            return DiagnosticReportListSerializer
        return DiagnosticReportDetailSerializer

    def perform_create(self, serializer):
        report = serializer.save(created_by=self.request.user)
        UserActivity.objects.create(
            user=self.request.user,
            action='report_generated',
            description=f'Создан отчет: {report.title}',
            metadata={'report_id': str(report.id)}
        )

class MaintenanceRecordViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'maintenance_type', 'status', 'assigned_to']
    search_fields = ['maintenance_type']
    ordering_fields = ['scheduled_date', 'created_at']
    ordering = ['-scheduled_date']

    def get_queryset(self):
        return MaintenanceRecord.objects.filter(system__owner=self.request.user)

    def get_serializer_class(self):
        if self.action == 'list':
            return MaintenanceRecordListSerializer
        return MaintenanceRecordDetailSerializer

class SystemComponentViewSet(viewsets.ModelViewSet):
    serializer_class = SystemComponentSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'component_type', 'status']
    search_fields = ['name']
    ordering_fields = ['name', 'condition_score', 'created_at']
    ordering = ['system', 'component_type', 'name']

    def get_queryset(self):
        return SystemComponent.objects.filter(system__owner=self.request.user)

class AlertViewSet(viewsets.ModelViewSet):
    serializer_class = AlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'alert_type', 'severity', 'status']
    search_fields = ['message']
    ordering_fields = ['created_at', 'severity']
    ordering = ['-created_at']

    def get_queryset(self):
        return Alert.objects.filter(system__owner=self.request.user)

    @action(detail=True, methods=['post'])
    def acknowledge(self, request, pk=None):
        alert = self.get_object()
        alert.acknowledge(request.user)
        return Response({'message': 'Оповещение принято'})

    @action(detail=True, methods=['post'])
    def resolve(self, request, pk=None):
        alert = self.get_object()
        notes = request.data.get('notes', '')
        alert.resolve(request.user, notes)
        return Response({'message': 'Оповещение решено'})
