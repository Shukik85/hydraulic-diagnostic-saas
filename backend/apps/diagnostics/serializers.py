from rest_framework import serializers
from django.utils import timezone
from datetime import timedelta
from .models import (
    HydraulicSystem, SensorData, SensorType, DiagnosticReport,
    MaintenanceRecord, SystemComponent, Alert
)

class HydraulicSystemListSerializer(serializers.ModelSerializer):
    """Сериализатор для списка систем"""
    health_score = serializers.SerializerMethodField()
    latest_activity = serializers.SerializerMethodField()
    critical_alerts_count = serializers.SerializerMethodField()
    system_type_display = serializers.CharField(source='get_system_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'id', 'name', 'system_type', 'system_type_display', 
            'status', 'status_display', 'location', 'criticality',
            'health_score', 'latest_activity', 'critical_alerts_count',
            'created_at', 'updated_at'
        ]
    
    def get_health_score(self, obj):
        return obj.get_health_score()
    
    def get_latest_activity(self, obj):
        latest_data = obj.sensordata_set.order_by('-timestamp').first()
        return latest_data.timestamp if latest_data else None
    
    def get_critical_alerts_count(self, obj):
        return obj.alert_set.filter(
            status='active',
            severity__in=['high', 'critical']
        ).count()

class HydraulicSystemDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор системы"""
    health_score = serializers.SerializerMethodField()
    latest_sensor_data = serializers.SerializerMethodField()
    recent_reports = serializers.SerializerMethodField()
    upcoming_maintenance = serializers.SerializerMethodField()
    system_type_display = serializers.CharField(source='get_system_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    criticality_display = serializers.CharField(source='get_criticality_display', read_only=True)
    owner_info = serializers.SerializerMethodField()
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'id', 'name', 'description', 'system_type', 'system_type_display',
            'manufacturer', 'model', 'serial_number', 'max_pressure', 'max_flow',
            'operating_temperature_min', 'operating_temperature_max', 'fluid_type',
            'status', 'status_display', 'criticality', 'criticality_display',
            'location', 'installation_date', 'last_maintenance', 'next_maintenance',
            'custom_parameters', 'owner_info', 'created_at', 'updated_at',
            'health_score', 'latest_sensor_data', 'recent_reports', 'upcoming_maintenance'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_health_score(self, obj):
        return obj.get_health_score()
    
    def get_latest_sensor_data(self, obj):
        return obj.get_latest_sensor_data()
    
    def get_recent_reports(self, obj):
        recent_reports = obj.diagnosticreport_set.order_by('-created_at')[:3]
        return DiagnosticReportListSerializer(recent_reports, many=True).data
    
    def get_upcoming_maintenance(self, obj):
        upcoming = obj.maintenancerecord_set.filter(
            status='planned',
            scheduled_date__gte=timezone.now()
        ).order_by('scheduled_date').first()
        return MaintenanceRecordListSerializer(upcoming).data if upcoming else None
    
    def get_owner_info(self, obj):
        return {
            'id': obj.owner.id,
            'username': obj.owner.username,
            'full_name': obj.owner.get_full_name() or obj.owner.username,
            'email': obj.owner.email
        }

class SensorTypeSerializer(serializers.ModelSerializer):
    """Сериализатор типов датчиков"""
    class Meta:
        model = SensorType
        fields = '__all__'

class SensorDataSerializer(serializers.ModelSerializer):
    """Сериализатор данных датчиков"""
    sensor_type_display = serializers.CharField(source='get_sensor_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    system_name = serializers.CharField(source='system.name', read_only=True)
    
    class Meta:
        model = SensorData
        fields = [
            'id', 'system', 'system_name', 'sensor_type', 'sensor_type_display',
            'sensor_id', 'value', 'unit', 'timestamp', 'status', 'status_display',
            'is_critical', 'warning_message', 'raw_data', 'quality_score',
            'deviation_from_normal', 'trend_direction', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class SensorDataBulkSerializer(serializers.Serializer):
    """Сериализатор для массовой загрузки данных датчиков"""
    sensor_data = SensorDataSerializer(many=True)
    
    def create(self, validated_data):
        sensor_data_list = validated_data['sensor_data']
        created_data = []
        
        for data in sensor_data_list:
            sensor_data = SensorData.objects.create(**data)
            created_data.append(sensor_data)
        
        return {'created_count': len(created_data), 'data': created_data}

class DiagnosticReportListSerializer(serializers.ModelSerializer):
    """Сериализатор списка отчетов диагностики"""
    system_name = serializers.CharField(source='system.name', read_only=True)
    report_type_display = serializers.CharField(source='get_report_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = DiagnosticReport
        fields = [
            'id', 'system', 'system_name', 'title', 'report_type', 'report_type_display',
            'severity', 'severity_display', 'status', 'status_display',
            'ai_confidence', 'created_at', 'completed_at'
        ]

class DiagnosticReportDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор отчета диагностики"""
    system_info = serializers.SerializerMethodField()
    created_by_info = serializers.SerializerMethodField()
    findings_count = serializers.SerializerMethodField()
    recommendations_count = serializers.SerializerMethodField()
    
    class Meta:
        model = DiagnosticReport
        fields = [
            'id', 'system', 'system_info', 'title', 'description',
            'report_type', 'severity', 'status', 'findings', 'recommendations',
            'analysis_data', 'ai_confidence', 'ai_analysis', 'attachments',
            'created_by', 'created_by_info', 'findings_count', 'recommendations_count',
            'created_at', 'completed_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_system_info(self, obj):
        return {
            'id': obj.system.id,
            'name': obj.system.name,
            'type': obj.system.get_system_type_display()
        }
    
    def get_created_by_info(self, obj):
        if obj.created_by:
            return {
                'id': obj.created_by.id,
                'username': obj.created_by.username,
                'full_name': obj.created_by.get_full_name() or obj.created_by.username
            }
        return None
    
    def get_findings_count(self, obj):
        return len(obj.findings) if obj.findings else 0
    
    def get_recommendations_count(self, obj):
        return len(obj.recommendations) if obj.recommendations else 0

class MaintenanceRecordListSerializer(serializers.ModelSerializer):
    """Сериализатор списка записей ТО"""
    system_name = serializers.CharField(source='system.name', read_only=True)
    maintenance_type_display = serializers.CharField(source='get_maintenance_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    assigned_to_name = serializers.CharField(source='assigned_to.get_full_name', read_only=True)
    is_overdue = serializers.SerializerMethodField()
    
    class Meta:
        model = MaintenanceRecord
        fields = [
            'id', 'system', 'system_name', 'title', 'maintenance_type', 
            'maintenance_type_display', 'status', 'status_display',
            'scheduled_date', 'assigned_to', 'assigned_to_name', 'is_overdue',
            'estimated_cost', 'actual_cost', 'created_at'
        ]
    
    def get_is_overdue(self, obj):
        return obj.is_overdue()

class MaintenanceRecordDetailSerializer(serializers.ModelSerializer):
    """Детальный сериализатор записи ТО"""
    system_info = serializers.SerializerMethodField()
    assigned_to_info = serializers.SerializerMethodField()
    completed_by_info = serializers.SerializerMethodField()
    duration = serializers.SerializerMethodField()
    
    class Meta:
        model = MaintenanceRecord
        fields = [
            'id', 'system', 'system_info', 'title', 'description',
            'maintenance_type', 'status', 'scheduled_date', 'started_at', 'completed_at',
            'estimated_duration', 'duration', 'assigned_to', 'assigned_to_info',
            'completed_by', 'completed_by_info', 'work_performed', 'parts_replaced',
            'materials_used', 'estimated_cost', 'actual_cost', 'success', 
            'notes', 'attachments', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_system_info(self, obj):
        return {
            'id': obj.system.id,
            'name': obj.system.name,
            'type': obj.system.get_system_type_display()
        }
    
    def get_assigned_to_info(self, obj):
        if obj.assigned_to:
            return {
                'id': obj.assigned_to.id,
                'username': obj.assigned_to.username,
                'full_name': obj.assigned_to.get_full_name() or obj.assigned_to.username
            }
        return None
    
    def get_completed_by_info(self, obj):
        if obj.completed_by:
            return {
                'id': obj.completed_by.id,
                'username': obj.completed_by.username,
                'full_name': obj.completed_by.get_full_name() or obj.completed_by.username
            }
        return None
    
    def get_duration(self, obj):
        if obj.started_at and obj.completed_at:
            duration = obj.completed_at - obj.started_at
            return str(duration)
        return None

class SystemComponentSerializer(serializers.ModelSerializer):
    """Сериализатор компонентов системы"""
    component_type_display = serializers.CharField(source='get_component_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    system_name = serializers.CharField(source='system.name', read_only=True)
    
    class Meta:
        model = SystemComponent
        fields = [
            'id', 'system', 'system_name', 'name', 'component_type', 
            'component_type_display', 'manufacturer', 'model', 'serial_number',
            'part_number', 'status', 'status_display', 'condition_score',
            'installation_date', 'last_inspection', 'next_maintenance',
            'specifications', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

class AlertSerializer(serializers.ModelSerializer):
    """Сериализатор оповещений"""
    system_name = serializers.CharField(source='system.name', read_only=True)
    alert_type_display = serializers.CharField(source='get_alert_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    acknowledged_by_name = serializers.CharField(source='acknowledged_by.get_full_name', read_only=True)
    resolved_by_name = serializers.CharField(source='resolved_by.get_full_name', read_only=True)
    
    class Meta:
        model = Alert
        fields = [
            'id', 'system', 'system_name', 'alert_type', 'alert_type_display',
            'severity', 'severity_display', 'status', 'status_display', 'title', 'message',
            'metadata', 'acknowledged_by', 'acknowledged_by_name', 'acknowledged_at',
            'resolved_by', 'resolved_by_name', 'resolved_at', 'resolution_notes',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class SystemStatsSerializer(serializers.Serializer):
    """Сериализатор статистики системы"""
    total_systems = serializers.IntegerField()
    active_systems = serializers.IntegerField()
    systems_in_maintenance = serializers.IntegerField()
    systems_with_alerts = serializers.IntegerField()
    total_sensor_readings_today = serializers.IntegerField()
    critical_alerts_today = serializers.IntegerField()
    reports_generated_this_month = serializers.IntegerField()
    average_health_score = serializers.FloatField()
    overdue_maintenance_count = serializers.IntegerField()

class DiagnosticRequestSerializer(serializers.Serializer):
    """Сериализатор запроса диагностики"""
    system_ids = serializers.ListField(
        child=serializers.UUIDField(),
        help_text="Список ID систем для диагностики"
    )
    report_type = serializers.ChoiceField(
        choices=DiagnosticReport.REPORT_TYPES,
        default='manual'
    )
    include_ai_analysis = serializers.BooleanField(default=True)
    time_range_hours = serializers.IntegerField(default=24, min_value=1, max_value=720)
    
    def validate_system_ids(self, value):
        # Проверка существования систем и доступа пользователя
        user = self.context['request'].user
        systems = HydraulicSystem.objects.filter(id__in=value, owner=user)
        
        if len(systems) != len(value):
            raise serializers.ValidationError("Некоторые системы не найдены или недоступны")
        
        return value
