"""
DRF Serializers for Sensor Data API.

Provides serializers for sensor nodes, configurations, and readings
with proper validation and nested relationships.
"""

from rest_framework import serializers
from django.utils import timezone
from datetime import timedelta

from .models import SensorNode, SensorConfig, SensorReading


class SensorConfigSerializer(serializers.ModelSerializer):
    """Serializer for sensor configuration."""
    
    class Meta:
        model = SensorConfig
        fields = [
            'id', 'name', 'register_address', 'data_type', 'unit',
            'scale_factor', 'offset', 'validation_min', 'validation_max',
            'description', 'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class SensorNodeSerializer(serializers.ModelSerializer):
    """Serializer for sensor nodes."""
    
    sensors = SensorConfigSerializer(many=True, read_only=True)
    sensors_count = serializers.SerializerMethodField()
    status_display = serializers.SerializerMethodField()
    
    class Meta:
        model = SensorNode
        fields = [
            'id', 'name', 'protocol', 'host_address', 'port',
            'protocol_config', 'is_active', 'connection_status',
            'last_poll_time', 'last_poll_success', 'last_error',
            'created_at', 'updated_at', 'sensors', 'sensors_count',
            'status_display'
        ]
        read_only_fields = [
            'id', 'connection_status', 'last_poll_time', 
            'last_poll_success', 'last_error', 'created_at', 'updated_at'
        ]
    
    def get_sensors_count(self, obj):
        """Get count of active sensors for this node."""
        return obj.sensors.filter(is_active=True).count()
    
    def get_status_display(self, obj):
        """Get human-readable status with additional context."""
        status = obj.connection_status
        
        if obj.last_poll_time:
            time_since_poll = timezone.now() - obj.last_poll_time
            if time_since_poll > timedelta(minutes=10):
                status = f"{status} (stale)"
            elif obj.last_poll_success:
                status = f"{status} (ok)"
        
        return status


class SensorReadingSerializer(serializers.ModelSerializer):
    """Serializer for sensor readings with related sensor info."""
    
    sensor_name = serializers.CharField(source='sensor_config.name', read_only=True)
    sensor_unit = serializers.CharField(source='sensor_config.unit', read_only=True)
    node_name = serializers.CharField(source='sensor_config.node.name', read_only=True)
    
    class Meta:
        model = SensorReading
        fields = [
            'id', 'sensor_config_id', 'timestamp', 'raw_value', 
            'processed_value', 'quality', 'error_message',
            'collection_latency_ms', 'sensor_name', 'sensor_unit',
            'node_name'
        ]
        read_only_fields = ['id']


class SensorReadingListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing many sensor readings."""
    
    sensor_name = serializers.CharField(source='sensor_config.name', read_only=True)
    
    class Meta:
        model = SensorReading
        fields = [
            'id', 'timestamp', 'processed_value', 'quality', 
            'sensor_name'
        ]


class SensorDataTimeSeriesSerializer(serializers.Serializer):
    """Serializer for time-series sensor data aggregation."""
    
    timestamp = serializers.DateTimeField()
    value = serializers.FloatField()
    quality = serializers.CharField(max_length=20)
    
    # Optional aggregation fields
    avg_value = serializers.FloatField(required=False)
    min_value = serializers.FloatField(required=False)
    max_value = serializers.FloatField(required=False)
    count = serializers.IntegerField(required=False)


class SensorStatsSerializer(serializers.Serializer):
    """Serializer for sensor statistics."""
    
    sensor_config_id = serializers.IntegerField()
    sensor_name = serializers.CharField()
    total_readings = serializers.IntegerField()
    good_readings = serializers.IntegerField()
    bad_readings = serializers.IntegerField()
    uncertain_readings = serializers.IntegerField()
    avg_value = serializers.FloatField(allow_null=True)
    min_value = serializers.FloatField(allow_null=True)
    max_value = serializers.FloatField(allow_null=True)
    last_reading_time = serializers.DateTimeField(allow_null=True)
    data_quality_percent = serializers.FloatField()


class NodeHealthSerializer(serializers.Serializer):
    """Serializer for node health status."""
    
    node_id = serializers.IntegerField()
    node_name = serializers.CharField()
    protocol = serializers.CharField()
    is_active = serializers.BooleanField()
    connection_status = serializers.CharField()
    last_poll_time = serializers.DateTimeField(allow_null=True)
    last_poll_success = serializers.BooleanField()
    sensors_count = serializers.IntegerField()
    recent_readings_count = serializers.IntegerField()
    avg_collection_latency_ms = serializers.FloatField(allow_null=True)
    error_rate_percent = serializers.FloatField()
