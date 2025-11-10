from rest_framework import serializers
from diagnostics.models import SensorData
from django.utils import timezone
import uuid

class SensorReadingSerializer(serializers.Serializer):
    sensor_id = serializers.UUIDField()
    timestamp = serializers.DateTimeField()
    value = serializers.FloatField()
    unit = serializers.ChoiceField(choices=["bar", "celsius", "rpm", "lpm"])
    quality = serializers.IntegerField(min_value=0, max_value=100, required=False, default=100)

    def validate_timestamp(self, value):
        if value > timezone.now() + timezone.timedelta(minutes=5):
            raise serializers.ValidationError("Timestamp cannot be more than 5 minutes in the future")
        if value < timezone.now() - timezone.timedelta(days=5*366):
            raise serializers.ValidationError("Timestamp too old (retention policy)")
        return value

class SensorBulkIngestSerializer(serializers.Serializer):
    system_id = serializers.UUIDField()
    readings = SensorReadingSerializer(many=True, min_length=1, max_length=10000)

    def validate(self, attrs):
        # Custom batch validation logic if needed
        return attrs
