from rest_framework import serializers
from .models import HydraulicSystem, SensorData, DiagnosticReport, Equipment, Diagnosis
from apps.users.models import User

class UserBasicSerializer(serializers.ModelSerializer):
    """Базовая информация о пользователе"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class SensorDataSerializer(serializers.ModelSerializer):
    """Сериализатор для данных датчиков"""
    sensor_type_display = serializers.CharField(
        source='get_sensor_type_display', read_only=True)
    
    class Meta:
        model = SensorData
        fields = [
            'id', 'sensor_type', 'sensor_type_display', 'value', 'unit',
            'timestamp', 'is_critical', 'warning_message'
        ]

class DiagnosticReportSerializer(serializers.ModelSerializer):
    """Сериализатор для отчетов диагностики"""
    severity_display = serializers.CharField(
        source='get_severity_display', read_only=True)
    
    class Meta:
        model = DiagnosticReport
        fields = [
            'id', 'title', 'description', 'severity', 'severity_display',
            'ai_analysis', 'recommendations', 'created_at', 'resolved_at'
        ]

class EquipmentSerializer(serializers.ModelSerializer):
    """Сериализатор для оборудования"""
    equipment_type_display = serializers.CharField(
        source='get_equipment_type_display', read_only=True)
    status_display = serializers.CharField(
        source='get_status_display', read_only=True)
    
    class Meta:
        model = Equipment
        fields = [
            'id', 'name', 'equipment_type', 'equipment_type_display',
            'model_number', 'serial_number', 'status', 'status_display',
            'installed_date', 'last_maintenance', 'next_maintenance',
            'created_at', 'updated_at'
        ]

class DiagnosisSerializer(serializers.ModelSerializer):
    """Сериализатор для диагностики"""
    status_display = serializers.CharField(
        source='get_status_display', read_only=True)
    priority_display = serializers.CharField(
        source='get_priority_display', read_only=True)
    assigned_to = UserBasicSerializer(read_only=True)
    created_by = UserBasicSerializer(read_only=True)
    equipment = EquipmentSerializer(read_only=True)
    
    class Meta:
        model = Diagnosis
        fields = [
            'id', 'system', 'equipment', 'title', 'description',
            'status', 'status_display', 'priority', 'priority_display',
            'findings', 'root_cause', 'solution',
            'ai_confidence', 'ai_suggestions',
            'assigned_to', 'created_by',
            'created_at', 'updated_at', 'completed_at'
        ]

class HydraulicSystemListSerializer(serializers.ModelSerializer):
    """Сериализатор для списка гидравлических систем"""
    system_type_display = serializers.CharField(
        source='get_system_type_display', read_only=True)
    status_display = serializers.CharField(
        source='get_status_display', read_only=True)
    owner = UserBasicSerializer(read_only=True)
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'id', 'name', 'system_type', 'system_type_display',
            'status', 'status_display', 'location', 'owner', 'created_at'
        ]

class HydraulicSystemDetailSerializer(serializers.ModelSerializer):
    """Подробный сериализатор для гидравлической системы"""
    system_type_display = serializers.CharField(
        source='get_system_type_display', read_only=True)
    status_display = serializers.CharField(
        source='get_status_display', read_only=True)
    owner = UserBasicSerializer(read_only=True)
    sensor_data = SensorDataSerializer(many=True, read_only=True)
    reports = DiagnosticReportSerializer(many=True, read_only=True)
    equipment = EquipmentSerializer(many=True, read_only=True)
    diagnoses = DiagnosisSerializer(many=True, read_only=True)
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'id', 'name', 'system_type', 'system_type_display',
            'location', 'status', 'status_display',
            'max_pressure', 'flow_rate', 'temperature_range',
            'owner', 'created_at', 'updated_at',
            'sensor_data', 'reports', 'equipment', 'diagnoses'
        ]

class HydraulicSystemCreateSerializer(serializers.ModelSerializer):
    """Сериализатор для создания гидравлической системы"""
    
    class Meta:
        model = HydraulicSystem
        fields = [
            'name', 'system_type', 'location', 'status',
            'max_pressure', 'flow_rate', 'temperature_range'
        ]
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)
