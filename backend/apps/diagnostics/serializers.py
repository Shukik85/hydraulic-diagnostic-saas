from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import serializers

from .models import (
    DiagnosticReport,
    HydraulicSystem,
    SensorData,
    SystemComponent,
)

User = get_user_model()


class ChoiceDisplayMixin:
    """Миксин для получения человекочитаемых значений полей выбора."""

    def get_choice_display(self, obj, field_name):
        method_name = f"get_{field_name}_display"
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                return method()
        return None


class UserBasicSerializer(serializers.ModelSerializer):
    """Базовый сериализатор пользователя."""

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name"]
        read_only_fields = fields


class EquipmentSerializer(serializers.Serializer):
    """
    Зависит от реальной модели Equipment.
    Здесь приведён шаблон, подставьте реальную модель и поля.
    """

    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=200)
    specification = serializers.JSONField(required=False)


class SensorDataSerializer(serializers.ModelSerializer):
    """Сериализатор данных датчиков."""

    class Meta:
        model = SensorData
        fields = ["id", "system", "component", "timestamp", "value", "unit"]
        read_only_fields = ["id"]
        extra_kwargs = {
            "timestamp": {"default": timezone.now},
        }


class SystemComponentSerializer(serializers.ModelSerializer):
    """Сериализатор компонентов системы."""

    class Meta:
        model = SystemComponent
        fields = ["id", "system", "name", "specification", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class HydraulicSystemListSerializer(ChoiceDisplayMixin, serializers.ModelSerializer):
    """Сериализатор для списка систем с отображениями полей выбора."""

    system_type_display = serializers.SerializerMethodField()
    status_display = serializers.SerializerMethodField()
    criticality_display = serializers.SerializerMethodField()
    latest_activity = serializers.SerializerMethodField()

    class Meta:
        model = HydraulicSystem
        fields = [
            "id",
            "name",
            "description",
            "system_type",
            "system_type_display",
            "status",
            "status_display",
            "criticality",
            "criticality_display",
            "created_at",
            "updated_at",
            "latest_activity",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "latest_activity",
        ]

    def get_system_type_display(self, obj):
        return self.get_choice_display(obj, "system_type")

    def get_status_display(self, obj):
        return self.get_choice_display(obj, "status")

    def get_criticality_display(self, obj):
        return self.get_choice_display(obj, "criticality")

    def get_latest_activity(self, obj):
        last = obj.sensor_data.order_by("-timestamp").first()
        return last.timestamp if last else None


class DiagnosticReportSerializer(ChoiceDisplayMixin, serializers.ModelSerializer):
    """Сериализатор отчетов диагностики."""

    system = HydraulicSystemListSerializer(read_only=True)
    created_by = UserBasicSerializer(read_only=True)
    severity_display = serializers.SerializerMethodField()
    status_display = serializers.SerializerMethodField()

    class Meta:
        model = DiagnosticReport
        fields = [
            "id",
            "system",
            "title",
            "severity",
            "severity_display",
            "status",
            "status_display",
            "ai_confidence",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "system",
            "severity_display",
            "status_display",
            "created_by",
            "created_at",
            "updated_at",
        ]

    def get_severity_display(self, obj):
        return self.get_choice_display(obj, "severity")

    def get_status_display(self, obj):
        return self.get_choice_display(obj, "status")


class DiagnosticEngineSettingsSerializer(serializers.Serializer):
    """
    Пример сериализатора настроек для DiagnosticEngine.
    Замените на реальные поля настроек при необходимости.
    """

    model_type = serializers.ChoiceField(choices=["isolation_forest", "random_forest"])
    threshold = serializers.FloatField(min_value=0, max_value=1)
