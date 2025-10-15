from rest_framework import serializers
from django.utils import timezone
from datetime import timedelta
from .models import (
    HydraulicSystem, SensorData, SensorType, DiagnosticReport,
    MaintenanceRecord, SystemComponent, Alert, Diagnosis, Equipment
)


class UserBasicSerializer(serializers.ModelSerializer):
    """Базовый сериализатор пользователя"""
    class Meta:
        model = 'auth.User'
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = fields


class EquipmentSerializer(serializers.ModelSerializer):
    """Сериализатор оборудования"""
    class Meta:
        model = Equipment
        fields = '__all__'


class SensorDataSerializer(serializers.ModelSerializer):
    """Сериализатор данных датчиков"""
    class Meta:
        model = SensorData
        fields = '__all__'


class HydraulicSystemListSerializer(serializers.ModelSerializer):
    """Сериализатор для списка систем"""
    health_score = serializers.SerializerMethodField()
    latest_activity = serializers.SerializerMethodField()
    critical_alerts_count = serializers.SerializerMethodField()
    system_type_display = serializers.CharField(
        source='get_system_type_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    
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
    system_type_display = serializers.CharField(
        source='get_system_type_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    criticality_display = serializers.CharField(
        source='get_criticality_display', read_only=True
    )
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
    
    def get_owner_info(self, obj):
        if obj.owner:
            return UserBasicSerializer(obj.owner).data
        return None
    
    def get_health_score(self, obj):
        return obj.get_health_score()
    
    def get_latest_sensor_data(self, obj):
        latest = obj.sensordata_set.order_by('-timestamp')[:10]
        return SensorDataSerializer(latest, many=True).data
    
    def get_recent_reports(self, obj):
        reports = obj.diagnosticreport_set.order_by('-generated_at')[:5]
        return DiagnosticReportSerializer(reports, many=True).data
    
    def get_upcoming_maintenance(self, obj):
        upcoming = obj.maintenancerecord_set.filter(
            scheduled_date__gte=timezone.now()
        ).order_by('scheduled_date')[:5]
        return MaintenanceRecordSerializer(upcoming, many=True).data


class MaintenanceRecordSerializer(serializers.ModelSerializer):
    """Сериализатор записей обслуживания"""
    system = HydraulicSystemListSerializer(read_only=True)
    performed_by = UserBasicSerializer(read_only=True)
    maintenance_type_display = serializers.CharField(
        source='get_maintenance_type_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    
    class Meta:
        model = MaintenanceRecord
        fields = [
            'id', 'system', 'maintenance_type', 'maintenance_type_display',
            'description', 'scheduled_date', 'completed_date', 'performed_by',
            'status', 'status_display', 'notes', 'cost', 'parts_replaced',
            'created_at', 'updated_at'
        ]


class DiagnosticReportSerializer(serializers.ModelSerializer):
    """Сериализатор диагностических отчетов"""
    system = HydraulicSystemListSerializer(read_only=True)
    generated_by = UserBasicSerializer(read_only=True)
    report_type_display = serializers.CharField(
        source='get_report_type_display', read_only=True
    )
    
    class Meta:
        model = DiagnosticReport
        fields = [
            'id', 'title', 'report_type', 'report_type_display',
            'system', 'generated_by', 'generated_at', 'data', 'file_path',
            'created_at', 'updated_at'
        ]


class DiagnosisSerializer(serializers.ModelSerializer):
    """Сериализатор для диагностики"""
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    severity_display = serializers.CharField(
        source='get_severity_display', read_only=True
    )
    assigned_to = UserBasicSerializer(read_only=True)
    created_by = UserBasicSerializer(read_only=True)
    equipment = EquipmentSerializer(read_only=True)
    report = DiagnosticReportSerializer(read_only=True)
    
    class Meta:
        model = Diagnosis
        fields = [
            'id', 'title', 'description', 'severity', 'severity_display',
            'status', 'status_display', 'equipment', 'assigned_to', 'created_by',
            'report', 'findings', 'recommendations', 'diagnosed_at', 'resolved_at'
        ]


class HydraulicSystemSerializer(serializers.ModelSerializer):
    """Сериализатор для гидравлической системы"""
    system_type_display = serializers.CharField(
        source='get_system_type_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    owner = UserBasicSerializer(read_only=True)
    equipment = EquipmentSerializer(many=True, read_only=True)
    sensor_data = SensorDataSerializer(many=True, read_only=True)
    recent_diagnoses = serializers.SerializerMethodField()
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'id', 'name', 'system_type', 'system_type_display', 'location',
            'status', 'status_display', 'max_pressure', 'flow_rate',
            'temperature_range', 'owner', 'equipment', 'sensor_data',
            'recent_diagnoses', 'created_at', 'updated_at'
        ]
    
    def get_recent_diagnoses(self, obj):
        recent = obj.diagnoses.all()[:5]
        return DiagnosisSerializer(recent, many=True).data


class AlertSerializer(serializers.ModelSerializer):
    """Сериализатор оповещений"""
    system = HydraulicSystemListSerializer(read_only=True)
    severity_display = serializers.CharField(
        source='get_severity_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    
    class Meta:
        model = Alert
        fields = [
            'id', 'system', 'title', 'message', 'severity', 'severity_display',
            'status', 'status_display', 'triggered_at', 'acknowledged_at',
            'resolved_at', 'created_at', 'updated_at'
        ]


class SystemComponentSerializer(serializers.ModelSerializer):
    """Сериализатор компонентов системы"""
    system = HydraulicSystemListSerializer(read_only=True)
    component_type_display = serializers.CharField(
        source='get_component_type_display', read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display', read_only=True
    )
    
    class Meta:
        model = SystemComponent
        fields = [
            'id', 'system', 'name', 'component_type', 'component_type_display',
            'manufacturer', 'model', 'serial_number', 'status', 'status_display',
            'installation_date', 'last_maintenance', 'next_maintenance',
            'specifications', 'created_at', 'updated_at'
        ]


class BulkOperationSerializer(serializers.Serializer):
    """Сериализатор для массовых операций"""
    system_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=True,
        help_text="Список ID систем для операции"
    )
    operation = serializers.ChoiceField(
        choices=['update_status', 'schedule_maintenance', 'generate_report'],
        required=True,
        help_text="Тип операции"
    )
    parameters = serializers.JSONField(
        required=False,
        help_text="Дополнительные параметры операции"
    )
    
    def validate_system_ids(self, value):
        # Проверка существования систем и доступа пользователя
        user = self.context['request'].user
        systems = HydraulicSystem.objects.filter(id__in=value, owner=user)
        
        if len(systems) != len(value):
            raise serializers.ValidationError("Некоторые системы не найдены или недоступны")
        
        return value
