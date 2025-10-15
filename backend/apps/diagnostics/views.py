from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Q, Avg
from django.utils import timezone
from datetime import timedelta
import pandas as pd
import logging

from .models import (
    HydraulicSystem, SensorData, SensorType, DiagnosticReport,
    MaintenanceRecord, SystemComponent, Alert
)
from .serializers import (
    HydraulicSystemListSerializer, HydraulicSystemDetailSerializer,
    SensorDataSerializer, SensorDataBulkSerializer, SensorTypeSerializer,
    DiagnosticReportListSerializer, DiagnosticReportDetailSerializer,
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
        return HydraulicSystemDetailSerializer
    
    def perform_create(self, serializer):
        system = serializer.save(owner=self.request.user)
        
        # Логирование создания системы
        UserActivity.objects.create(
            user=self.request.user,
            action='system_created',
            description=f'Создана система: {system.name}',
            metadata={'system_id': str(system.id)}
        )
    
    def perform_update(self, serializer):
        system = serializer.save()
        
        # Логирование обновления системы
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
        
        # Логирование удаления системы
        UserActivity.objects.create(
            user=self.request.user,
            action='system_deleted',
            description=f'Удалена система: {system_name}',
            metadata={'system_id': system_id}
        )
    
    @action(detail=True, methods=['get'])
    def health_score(self, request, pk=None):
        """Получить индекс здоровья системы"""
        system = self.get_object()
        health_score = system.get_health_score()
        
        return Response({
            'system_id': system.id,
            'health_score': health_score,
            'status': self._get_health_status(health_score),
            'calculated_at': timezone.now()
        })
    
    @action(detail=True, methods=['get'])
    def latest_sensor_data(self, request, pk=None):
        """Получить последние данные датчиков системы"""
        system = self.get_object()
        latest_data = system.get_latest_sensor_data()
        
        return Response({
            'system_id': system.id,
            'sensor_data': latest_data,
            'retrieved_at': timezone.now()
        })
    
    @action(detail=True, methods=['get'])
    def diagnostics_history(self, request, pk=None):
        """История диагностики системы"""
        system = self.get_object()
        
        reports = DiagnosticReport.objects.filter(system=system).order_by('-created_at')[:10]
        serializer = DiagnosticReportListSerializer(reports, many=True)
        
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def run_diagnostic(self, request, pk=None):
        """Запуск диагностики системы"""
        system = self.get_object()
        
        try:
            # Получение данных для анализа
            sensor_data = self._get_system_sensor_data(system, hours=24)
            
            # Запуск AI анализа
            ai_engine = get_ai_engine()
            
            system_data = {
                'system_id': str(system.id),
                'sensor_data': sensor_data,
                'system_info': {
                    'type': system.system_type,
                    'age_months': self._calculate_system_age(system),
                    'maintenance_history': self._get_maintenance_history(system)
                }
            }
            
            # Обнаружение аномалий
            anomalies = ai_engine.detect_anomalies(pd.DataFrame(sensor_data))
            
            # Предсказание отказов
            failure_prediction = ai_engine.predict_failure_probability(system_data)
            
            # Генерация инсайтов
            insights = ai_engine.generate_diagnostic_insights(system_data)
            
            # Создание отчета диагностики
            report = DiagnosticReport.objects.create(
                system=system,
                title=f'Автоматическая диагностика {system.name}',
                description='Автоматический анализ системы с использованием AI',
                report_type='automated',
                findings=anomalies,
                recommendations=failure_prediction.get('recommendations', []),
                analysis_data=insights,
                ai_confidence=failure_prediction.get('failure_probability', 0),
                ai_analysis=insights.get('summary', ''),
                status='completed',
                completed_at=timezone.now()
            )
            
            # Логирование запуска диагностики
            UserActivity.objects.create(
                user=request.user,
                action='diagnostic_run',
                description=f'Запущена диагностика системы: {system.name}',
                metadata={
                    'system_id': str(system.id),
                    'report_id': str(report.id),
                    'anomalies_count': len(anomalies)
                }
            )
            
            return Response({
                'message': 'Диагностика завершена успешно',
                'report_id': report.id,
                'anomalies_found': len(anomalies),
                'failure_probability': failure_prediction.get('failure_probability', 0),
                'health_score': insights.get('system_health_score', 0)
            })
            
        except Exception as e:
            logger.error(f"Ошибка запуска диагностики для системы {system.id}: {e}")
            return Response({
                'error': 'Ошибка выполнения диагностики',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_health_status(self, score):
        """Определить статус здоровья по оценке"""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _get_system_sensor_data(self, system, hours=24):
        """Получить данные датчиков системы за указанный период"""
        cutoff_time = timezone.now() - timedelta(hours=hours)
        
        sensor_data = SensorData.objects.filter(
            system=system,
            timestamp__gte=cutoff_time
        ).values(
            'sensor_type', 'value', 'unit', 'timestamp', 
            'is_critical', 'status', 'quality_score'
        )
        
        return list(sensor_data)
    
    def _calculate_system_age(self, system):
        """Рассчитать возраст системы в месяцах"""
        if system.installation_date:
            age = timezone.now().date() - system.installation_date
            return age.days // 30
        return 0
    
    def _get_maintenance_history(self, system):
        """Получить историю обслуживания системы"""
        maintenance = MaintenanceRecord.objects.filter(
            system=system,
            status='completed'
        ).order_by('-completed_at')[:5]
        
        return [
            {
                'type': record.maintenance_type,
                'date': record.completed_at,
                'cost': float(record.actual_cost) if record.actual_cost else 0
            }
            for record in maintenance
        ]

class SensorDataViewSet(viewsets.ModelViewSet):
    """ViewSet для данных датчиков"""
    serializer_class = SensorDataSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['sensor_type', 'is_critical', 'status']
    ordering_fields = ['timestamp', 'value']
    ordering = ['-timestamp']
    
    def get_queryset(self):
        return SensorData.objects.filter(system__owner=self.request.user)
    
    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """Массовое создание данных датчиков"""
        serializer = SensorDataBulkSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        result = serializer.save()
        
        return Response({
            'message': f'Создано {result["created_count"]} записей',
            'created_count': result['created_count']
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['get'])
    def critical_alerts(self, request):
        """Получить критические оповещения"""
        critical_data = self.get_queryset().filter(
            is_critical=True,
            timestamp__gte=timezone.now() - timedelta(hours=24)
        )
        
        serializer = self.get_serializer(critical_data, many=True)
        return Response(serializer.data)

class DiagnosticReportViewSet(viewsets.ModelViewSet):
    """ViewSet для отчетов диагностики"""
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'report_type', 'severity', 'status']
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
        
        # Логирование создания отчета
        UserActivity.objects.create(
            user=self.request.user,
            action='report_generated',
            description=f'Создан отчет: {report.title}',
            metadata={'report_id': str(report.id)}
        )

class MaintenanceRecordViewSet(viewsets.ModelViewSet):
    """ViewSet для записей технического обслуживания"""
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'maintenance_type', 'status', 'assigned_to']
    ordering_fields = ['scheduled_date', 'created_at']
    ordering = ['-scheduled_date']
    
    def get_queryset(self):
        return MaintenanceRecord.objects.filter(system__owner=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'list':
            return MaintenanceRecordListSerializer
        return MaintenanceRecordDetailSerializer
    
    @action(detail=False, methods=['get'])
    def overdue(self, request):
        """Получить просроченные записи ТО"""
        overdue_records = self.get_queryset().filter(
            status='planned',
            scheduled_date__lt=timezone.now()
        )
        
        serializer = MaintenanceRecordListSerializer(overdue_records, many=True)
        return Response(serializer.data)

class SystemComponentViewSet(viewsets.ModelViewSet):
    """ViewSet для компонентов системы"""
    serializer_class = SystemComponentSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'component_type', 'status']
    ordering_fields = ['name', 'condition_score', 'created_at']
    ordering = ['system', 'component_type', 'name']
    
    def get_queryset(self):
        return SystemComponent.objects.filter(system__owner=self.request.user)

class AlertViewSet(viewsets.ModelViewSet):
    """ViewSet для оповещений"""
    serializer_class = AlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['system', 'alert_type', 'severity', 'status']
    ordering_fields = ['created_at', 'severity']
    ordering = ['-created_at']
    
    def get_queryset(self):
        return Alert.objects.filter(system__owner=self.request.user)
    
    @action(detail=True, methods=['post'])
    def acknowledge(self, request, pk=None):
        """Принять оповещение"""
        alert = self.get_object()
        alert.acknowledge(request.user)
        
        return Response({'message': 'Оповещение принято'})
    
    @action(detail=True, methods=['post'])
    def resolve(self, request, pk=None):
        """Решить оповещение"""
        alert = self.get_object()
        notes = request.data.get('notes', '')
        alert.resolve(request.user, notes)
        
        return Response({'message': 'Оповещение решено'})

class SensorTypeViewSet(viewsets.ModelViewSet):
    """ViewSet для типов датчиков"""
    queryset = SensorType.objects.filter(is_active=True)
    serializer_class = SensorTypeSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = StandardResultsSetPagination

@action(detail=False, methods=['get'])
def system_stats(request):
    """Получить статистику по системам пользователя"""
    user = request.user
    
    # Базовая статистика
    total_systems = HydraulicSystem.objects.filter(owner=user).count()
    active_systems = HydraulicSystem.objects.filter(owner=user, status='active').count()
    systems_in_maintenance = HydraulicSystem.objects.filter(
        owner=user, status='maintenance'
    ).count()
    
    # Системы с активными оповещениями
    systems_with_alerts = Alert.objects.filter(
        system__owner=user, status='active'
    ).values('system').distinct().count()
    
    # Показания датчиков за сегодня
    today = timezone.now().date()
    total_sensor_readings_today = SensorData.objects.filter(
        system__owner=user,
        timestamp__date=today
    ).count()
    
    # Критические оповещения за сегодня
    critical_alerts_today = SensorData.objects.filter(
        system__owner=user,
        is_critical=True,
        timestamp__date=today
    ).count()
    
    # Отчеты за этот месяц
    month_start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    reports_generated_this_month = DiagnosticReport.objects.filter(
        system__owner=user,
        created_at__gte=month_start
    ).count()
    
    # Средний индекс здоровья
    user_systems = HydraulicSystem.objects.filter(owner=user)
    health_scores = [system.get_health_score() for system in user_systems]
    average_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
    
    # Просроченное ТО
    overdue_maintenance_count = MaintenanceRecord.objects.filter(
        system__owner=user,
        status='planned',
        scheduled_date__lt=timezone.now()
    ).count()
    
    stats_data = {
        'total_systems': total_systems,
        'active_systems': active_systems,
        'systems_in_maintenance': systems_in_maintenance,
        'systems_with_alerts': systems_with_alerts,
        'total_sensor_readings_today': total_sensor_readings_today,
        'critical_alerts_today': critical_alerts_today,
        'reports_generated_this_month': reports_generated_this_month,
        'average_health_score': round(average_health_score, 1),
        'overdue_maintenance_count': overdue_maintenance_count,
    }
    
    serializer = SystemStatsSerializer(stats_data)
    return Response(serializer.data)
