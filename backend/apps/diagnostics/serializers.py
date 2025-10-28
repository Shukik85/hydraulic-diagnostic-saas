"""Модуль проекта с автогенерированным докстрингом."""

import re
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import serializers

from .models import (
    DiagnosticReport,
    HydraulicSystem,
    IntegratedDiagnosticResult,
    MathematicalModelResult,
    PhasePortraitResult,
    SensorData,
    SystemComponent,
    TribodiagnosticResult,
)

User = get_user_model()


class ChoiceDisplayMixin:
    """Миксин для получения человекочитаемых значений полей выбора."""

    def get_choice_display(self, obj, field_name):
        """Получает choice display.

        Args:
            obj (Any): Параметр obj
            field_name (Any): Параметр field_name

        """
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
    """Зависит от реальной модели Equipment.
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
        """Получает system type display.

        Args:
            obj (Any): Параметр obj

        """
        return self.get_choice_display(obj, "system_type")

    def get_status_display(self, obj):
        """Получает status display.

        Args:
            obj (Any): Параметр obj

        """
        return self.get_choice_display(obj, "status")

    def get_criticality_display(self, obj):
        """Получает criticality display.

        Args:
            obj (Any): Параметр obj

        """
        return self.get_choice_display(obj, "criticality")

    def get_latest_activity(self, obj):
        """Получает latest activity.

        Args:
            obj (Any): Параметр obj

        """
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
        """Получает severity display.

        Args:
            obj (Any): Параметр obj

        """
        return self.get_choice_display(obj, "severity")

    def get_status_display(self, obj):
        """Получает status display.

        Args:
            obj (Any): Параметр obj

        """
        return self.get_choice_display(obj, "status")


class DiagnosticEngineSettingsSerializer(serializers.Serializer):
    """Пример сериализатора настроек для DiagnosticEngine.
    Замените на реальные поля настроек при необходимости.
    """

    model_type = serializers.ChoiceField(choices=["isolation_forest", "random_forest"])
    threshold = serializers.FloatField(min_value=0, max_value=1)


# Диагностические сериализаторы для новых моделей Sprint 1


class MathematicalModelResultSerializer(serializers.ModelSerializer):
    """Сериализатор результатов математической модели.
    
    Включает валидацию отклонений >= 0 и соответствия статуса.
    """

    class Meta:
        model = MathematicalModelResult
        fields = [
            "id",
            "system",
            "analysis_date",
            "pressure_deviation",
            "flow_deviation",
            "speed_deviation",
            "max_deviation", 
            "mathematical_score",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def validate_pressure_deviation(self, value):
        """Валидация отклонения давления."""
        if value < 0:
            raise serializers.ValidationError("Отклонение давления не может быть отрицательным")
        return value

    def validate_flow_deviation(self, value):
        """Валидация отклонения расхода."""
        if value < 0:
            raise serializers.ValidationError("Отклонение расхода не может быть отрицательным")
        return value

    def validate_speed_deviation(self, value):
        """Валидация отклонения скорости."""
        if value < 0:
            raise serializers.ValidationError("Отклонение скорости не может быть отрицательным")
        return value

    def validate_mathematical_score(self, value):
        """Валидация математического скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Математический скор должен быть в диапазоне [0, 1]")
        return value

    def validate(self, attrs):
        """Комплексная валидация."""
        # Проверим что max_deviation соответствует максимуму из трех отклонений
        if all(k in attrs for k in ['pressure_deviation', 'flow_deviation', 'speed_deviation']):
            calculated_max = max(
                attrs['pressure_deviation'],
                attrs['flow_deviation'], 
                attrs['speed_deviation']
            )
            if 'max_deviation' in attrs and abs(attrs['max_deviation'] - calculated_max) > 0.01:
                raise serializers.ValidationError(
                    "max_deviation должен равняться максимуму из трех отклонений"
                )
        return attrs


class PhasePortraitResultSerializer(serializers.ModelSerializer):
    """Сериализатор результатов фазового портрета.
    
    Включает валидацию типа портрета и area_deviation >= 0.
    """

    class Meta:
        model = PhasePortraitResult
        fields = [
            "id",
            "system", 
            "analysis_date",
            "portrait_type",
            "area_deviation",
            "phase_score",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def validate_portrait_type(self, value):
        """Валидация типа фазового портрета."""
        valid_types = ['velocity_position', 'force_velocity', 'pressure_flow']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Тип портрета должен быть одним из: {', '.join(valid_types)}"
            )
        return value

    def validate_area_deviation(self, value):
        """Валидация отклонения площади."""
        if value < 0:
            raise serializers.ValidationError("Отклонение площади не может быть отрицательным")
        return value

    def validate_phase_score(self, value):
        """Валидация фазового скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Фазовый скор должен быть в диапазоне [0, 1]")
        return value

    def validate(self, attrs):
        """Комплексная валидация согласованности статуса с area_deviation."""
        if 'area_deviation' in attrs and 'status' in attrs:
            area_dev = attrs['area_deviation']
            status = attrs['status']
            
            # Проверка согласованности статуса с порогами
            if area_dev < 10 and status != 'normal':
                raise serializers.ValidationError(
                    "При отклонении площади < 10% статус должен быть 'normal'"
                )
            elif 10 <= area_dev < 25 and status != 'pre_fault':
                raise serializers.ValidationError(
                    "При отклонении площади 10-25% статус должен быть 'pre_fault'"
                )
            elif area_dev >= 25 and status != 'fault':
                raise serializers.ValidationError(
                    "При отклонении площади >= 25% статус должен быть 'fault'"
                )
        return attrs


class TribodiagnosticResultSerializer(serializers.ModelSerializer):
    """Сериализатор результатов трибодиагностики.
    
    Включает валидацию pH [0,14], формата ISO класса и числовых полей >= 0.
    """

    class Meta:
        model = TribodiagnosticResult
        fields = [
            "id",
            "system",
            "analysis_date",
            "iso_class",
            "particles_4_micron",
            "particles_6_micron", 
            "particles_14_micron",
            "water_content_ppm",
            "viscosity",
            "ph_level",
            "iron_content",
            "copper_content",
            "aluminum_content",
            "tribo_score",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def validate_iso_class(self, value):
        """Валидация формата ISO класса."""
        if not re.match(r'^\d{2}/\d{2}/\d{2}$', value):
            raise serializers.ValidationError(
                "ISO класс должен быть в формате 'NN/NN/NN' (например, '18/16/13')"
            )
        return value

    def validate_ph_level(self, value):
        """Валидация уровня pH."""
        if not (0 <= value <= 14):
            raise serializers.ValidationError("pH уровень должен быть в диапазоне [0, 14]")
        return value

    def validate_particles_4_micron(self, value):
        """Валидация частиц 4 мкм."""
        if value < 0:
            raise serializers.ValidationError("Количество частиц не может быть отрицательным")
        return value

    def validate_particles_6_micron(self, value):
        """Валидация частиц 6 мкм."""
        if value < 0:
            raise serializers.ValidationError("Количество частиц не может быть отрицательным")
        return value

    def validate_particles_14_micron(self, value):
        """Валидация частиц 14 мкм."""
        if value < 0:
            raise serializers.ValidationError("Количество частиц не может быть отрицательным")
        return value

    def validate_water_content_ppm(self, value):
        """Валидация содержания воды."""
        if value < 0:
            raise serializers.ValidationError("Содержание воды не может быть отрицательным")
        return value

    def validate_viscosity(self, value):
        """Валидация вязкости."""
        if value < 0:
            raise serializers.ValidationError("Вязкость не может быть отрицательной")
        return value

    def validate_iron_content(self, value):
        """Валидация содержания железа."""
        if value < 0:
            raise serializers.ValidationError("Содержание железа не может быть отрицательным")
        return value

    def validate_copper_content(self, value):
        """Валидация содержания меди."""
        if value < 0:
            raise serializers.ValidationError("Содержание меди не может быть отрицательным")
        return value

    def validate_aluminum_content(self, value):
        """Валидация содержания алюминия."""
        if value < 0:
            raise serializers.ValidationError("Содержание алюминия не может быть отрицательным")
        return value

    def validate_tribo_score(self, value):
        """Валидация трибологического скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Трибологический скор должен быть в диапазоне [0, 1]")
        return value


class IntegratedDiagnosticResultSerializer(serializers.ModelSerializer):
    """Сериализатор интегрированных результатов диагностики.
    
    integrated_score и overall_status вычисляются автоматически (read-only).
    """

    class Meta:
        model = IntegratedDiagnosticResult
        fields = [
            "id",
            "system",
            "analysis_date",
            "math_result",
            "phase_result",
            "tribo_result",
            "math_score",
            "phase_score", 
            "tribo_score",
            "integrated_score",
            "overall_status",
            "predicted_remaining_life",
            "confidence_level",
            "recommendations",
            "priority_actions",
            "created_at",
        ]
        read_only_fields = ["id", "integrated_score", "overall_status", "created_at"]

    def validate_math_score(self, value):
        """Валидация математического скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Математический скор должен быть в диапазоне [0, 1]")
        return value

    def validate_phase_score(self, value):
        """Валидация фазового скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Фазовый скор должен быть в диапазоне [0, 1]")
        return value

    def validate_tribo_score(self, value):
        """Валидация трибологического скора."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Трибологический скор должен быть в диапазоне [0, 1]")
        return value

    def validate_confidence_level(self, value):
        """Валидация уровня доверия."""
        if not (0 <= value <= 1):
            raise serializers.ValidationError("Уровень доверия должен быть в диапазоне [0, 1]")
        return value

    def validate_predicted_remaining_life(self, value):
        """Валидация предсказанного остаточного ресурса."""
        if value < 0:
            raise serializers.ValidationError("Остаточный ресурс не может быть отрицательным")
        return value
