import uuid
from decimal import Decimal

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex, GinIndex
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.db.models import Q
from django.db.models.functions import TruncDay
from django.utils import timezone


# ------------------------ QuerySets & Managers ------------------------ #


class HydraulicSystemQuerySet(models.QuerySet):
    def with_owner(self):
        return self.select_related("owner")

    def with_components(self):
        return self.prefetch_related("components")

    def active(self):
        return self.filter(status="active")

    def for_owner(self, owner_id):
        return self.filter(owner_id=owner_id)

    def with_prefetch(self):
        return self.prefetch_related(
            models.Prefetch(
                "components",
                queryset=SystemComponent.objects.only("id", "system_id", "name"),
            )
        )


class SystemComponentQuerySet(models.QuerySet):
    def with_system(self):
        return self.select_related("system")

    def for_system(self, system_id):
        return self.filter(system_id=system_id)


class SensorDataQuerySet(models.QuerySet):
    def for_system(self, system_id):
        return self.filter(system_id=system_id).select_related("component")

    def for_component(self, component_id):
        return self.filter(component_id=component_id)

    def time_range(self, start, end):
        return self.filter(timestamp__gte=start, timestamp__lt=end)

    def recent(self, hours=24):
        return self.filter(
            timestamp__gte=timezone.now() - timezone.timedelta(hours=hours)
        )

    def with_system_component(self):
        return self.select_related("system", "component")

    def critical(self):
        return self.filter(is_critical=True)

    def by_sensor_type(self, sensor_type):
        return self.filter(sensor_type=sensor_type)

    def recent_for_system(self, system_id, limit=1000):
        return (
            self.filter(system_id=system_id)
            .select_related("component")
            .only("timestamp", "value", "unit", "component_id", "sensor_type")
            .order_by("-timestamp")[:limit]
        )

    def for_component_range(self, component_id, ts_from, ts_to):
        return (
            self.filter(component_id=component_id, timestamp__range=(ts_from, ts_to))
            .only("timestamp", "value", "unit", "sensor_type")
            .order_by("timestamp")
        )


class DiagnosticReportQuerySet(models.QuerySet):
    def with_system(self):
        return self.select_related("system")

    def critical(self):
        return self.filter(severity__in=["error", "critical"])

    def open_critical(self):
        return self.filter(status="open", severity="critical")

    def recent_for_system(self, system_id, limit=100):
        return (
            self.filter(system_id=system_id)
            .only("id", "title", "severity", "status", "created_at", "ai_confidence")
            .order_by("-created_at")[:limit]
        )


# ------------------------------- Models ------------------------------- #


class HydraulicSystem(models.Model):
    """Гидравлическая система (оптимизировано под быстрые выборки)."""

    SYSTEM_TYPES = [
        ("industrial", "Промышленная"),
        ("mobile", "Мобильная"),
        ("marine", "Морская"),
        ("aviation", "Авиационная"),
        ("construction", "Строительная"),
        ("mining", "Горнодобывающая"),
        ("agricultural", "Сельскохозяйственная"),
    ]
    STATUS_CHOICES = [
        ("active", "Активна"),
        ("maintenance", "На обслуживании"),
        ("inactive", "Неактивна"),
        ("emergency", "Аварийная"),
        ("decommissioned", "Списана"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, db_index=True)
    description = models.TextField(blank=True, default="")
    system_type = models.CharField(max_length=50, choices=SYSTEM_TYPES, db_index=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="active", db_index=True
    )

    # Связь с пользователем
    owner = models.ForeignKey(
        "users.User",
        on_delete=models.PROTECT,
        related_name="hydraulic_systems",
        db_index=True,
    )

    criticality = models.CharField(max_length=20, default="medium", db_index=True)
    location = models.CharField(max_length=200, blank=True, default="")
    installation_date = models.DateField(null=True, blank=True)

    # Денормализация для быстрых ответов API
    components_count = models.PositiveIntegerField(default=0)
    last_reading_at = models.DateTimeField(null=True, blank=True, db_index=True)

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    objects = HydraulicSystemQuerySet.as_manager()

    class Meta:
        db_table = "diagnostics_hydraulicsystem"
        ordering = ["-updated_at"]
        indexes = [
            # Составные индексы для частых запросов
            BTreeIndex(fields=["owner", "status"], name="idx_hs_owner_status"),
            BTreeIndex(fields=["system_type", "status"], name="idx_hs_type_status"),
            BTreeIndex(fields=["owner", "system_type"], name="idx_hs_owner_type"),
            # BRIN индексы для временных полей (эффективны для больших объёмов)
            BrinIndex(
                fields=["created_at"], autosummarize=True, name="brin_hs_created"
            ),
            BrinIndex(
                fields=["updated_at"], autosummarize=True, name="brin_hs_updated"
            ),
            BrinIndex(
                fields=["last_reading_at"],
                autosummarize=True,
                name="brin_hs_last_reading",
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"], name="uq_hydraulicsystem_owner_name"
            ),
            models.CheckConstraint(
                check=~Q(name=""), name="chk_hs_name_not_empty"
            ),
        ]

    def clean(self):
        if not self.name.strip():
            raise ValidationError("System name cannot be empty")
        if self.status == "inactive" and self.components_count > 0:
            raise ValidationError("Inactive system cannot have components_count > 0")

    def __str__(self):
        return f"{self.name} ({self.system_type})"


class SystemComponent(models.Model):
    """Компонент гидравлической системы."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        related_name="components",
        on_delete=models.CASCADE,
        db_index=True,
    )
    name = models.CharField(max_length=255)
    specification = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = SystemComponentQuerySet.as_manager()

    class Meta:
        db_table = "diagnostics_systemcomponent"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["system", "name"], name="uq_systemcomponent_system_name"
            ),
        ]
        indexes = [
            BTreeIndex(fields=["system", "name"], name="idx_comp_system_name"),
            BrinIndex(fields=["created_at"], autosummarize=True, name="brin_comp_created"),
        ]

    def clean(self):
        if not self.name.strip():
            raise ValidationError("Component name cannot be empty")

    def __str__(self):
        return f"{self.system.name}::{self.name}"


class SensorData(models.Model):
    """Высокочастотные IoT данные датчиков (TimescaleDB hypertable)."""

    SENSOR_TYPES = [
        ("pressure", "Давление"),
        ("temperature", "Температура"),
        ("flow", "Поток"),
        ("vibration", "Вибрация"),
        ("level", "Уровень"),
        ("position", "Позиция"),
        ("speed", "Скорость"),
        ("torque", "Крутящий момент"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="sensor_data",
        db_index=True,
    )
    component = models.ForeignKey(
        SystemComponent,
        on_delete=models.SET_NULL,  # сохраняем исторические данные при удалении компонента
        null=True,
        blank=True,
        related_name="sensor_data",
        db_index=True,
    )

    timestamp = models.DateTimeField(db_index=True)
    sensor_type = models.CharField(max_length=64, choices=SENSOR_TYPES, db_index=True)
    value = models.FloatField(validators=[MinValueValidator(float("-inf"))])
    unit = models.CharField(max_length=32, default="", blank=True)

    # Критичность показания
    is_critical = models.BooleanField(default=False, db_index=True)
    warning_message = models.CharField(max_length=240, default="", blank=True)

    # Для точных агрегаций (опционально)
    value_decimal = models.DecimalField(
        max_digits=18, decimal_places=6, null=True, blank=True
    )

    # Сгенерированное поле для агрегатов по дням
    day_bucket = models.GeneratedField(
        expression=TruncDay("timestamp"),
        output_field=models.DateField(),
        db_persist=True,
        db_index=True,
    )

    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    objects = SensorDataQuerySet.as_manager()

    class Meta:
        db_table = "diagnostics_sensordata"
        ordering = ["-timestamp"]
        indexes = [
            # Основные составные индексы для частых запросов
            BTreeIndex(fields=["system", "timestamp"], name="idx_sd_system_ts"),
            BTreeIndex(fields=["component", "timestamp"], name="idx_sd_component_ts"),
            BTreeIndex(fields=["sensor_type", "timestamp"], name="idx_sd_type_ts"),
            BTreeIndex(
                fields=["system", "sensor_type", "timestamp"], name="idx_sd_sys_type_ts"
            ),
            BTreeIndex(fields=["is_critical", "timestamp"], name="idx_sd_critical_ts"),
            # BRIN индексы для временных полей (оптимальны для TimescaleDB)
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_sd_ts"),
            BrinIndex(
                fields=["created_at"], autosummarize=True, name="brin_sd_created"
            ),
            # Индекс для day_bucket агрегатов
            BTreeIndex(fields=["day_bucket"], name="idx_sd_day_bucket"),
        ]
        constraints = [
            models.CheckConstraint(
                check=Q(value__isnull=False) | Q(value_decimal__isnull=False),
                name="chk_sd_value_present",
            ),
        ]

    def clean(self):
        if self.timestamp and self.timestamp > timezone.now() + timezone.timedelta(
            minutes=5
        ):
            raise ValidationError("Timestamp cannot be more than 5 minutes in the future")
        if self.value is None and self.value_decimal is None:
            raise ValidationError("Either value or value_decimal must be provided")
        if self.unit and len(self.unit) > 32:
            raise ValidationError("Unit is too long (max 32 characters)")

    def save(self, *args, **kwargs):
        # Валидация перед сохранением
        self.full_clean()
        super().save(*args, **kwargs)

        # Обновляем last_reading_at у системы (без дополнительных запросов)
        if self.system_id and self.timestamp:
            HydraulicSystem.objects.filter(id=self.system_id).update(
                last_reading_at=self.timestamp
            )

    def __str__(self):
        return f"{self.sensor_type}@{self.system.name}:{self.component.name if self.component else 'N/A'}"


class DiagnosticReport(models.Model):
    """Диагностический отчёт с AI-анализом."""

    SEVERITY_CHOICES = [
        ("info", "Информация"),
        ("warning", "Предупреждение"),
        ("error", "Ошибка"),
        ("critical", "Критическая"),
    ]
    STATUS_CHOICES = [
        ("open", "Открыт"),
        ("in_progress", "В процессе"),
        ("closed", "Закрыт"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="diagnostic_reports",
        db_index=True,
    )
    title = models.CharField(max_length=255, db_index=True)
    severity = models.CharField(max_length=16, choices=SEVERITY_CHOICES, db_index=True)
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default="open", db_index=True
    )

    # AI уверенность в диапазоне 0.0-1.0
    ai_confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0
    )

    # Дополнительные метрики
    impacted_components_count = models.PositiveIntegerField(default=0)
    description = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = DiagnosticReportQuerySet.as_manager()

    class Meta:
        db_table = "diagnostics_diagnosticreport"
        ordering = ["-created_at"]
        indexes = [
            BTreeIndex(fields=["system", "created_at"], name="idx_dr_system_created"),
            BTreeIndex(fields=["severity", "created_at"], name="idx_dr_severity_created"),
            BTreeIndex(fields=["status", "severity"], name="idx_dr_status_severity"),
            BrinIndex(
                fields=["created_at"], autosummarize=True, name="brin_dr_created"
            ),
        ]
        constraints = [
            models.CheckConstraint(
                check=Q(ai_confidence__gte=0.0) & Q(ai_confidence__lte=1.0),
                name="chk_dr_ai_confidence_range",
            ),
        ]

    def clean(self):
        if self.ai_confidence < 0.0 or self.ai_confidence > 1.0:
            raise ValidationError("AI confidence must be between 0.0 and 1.0")

    def __str__(self):
        return f"{self.title} ({self.severity}/{self.status})"