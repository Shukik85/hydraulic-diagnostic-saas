import uuid
from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.db.models.functions import TruncDay
from django.utils import timezone

# ------------------------ QuerySets & Managers ------------------------ #


class HydraulicSystemQuerySet(models.QuerySet):
    def with_prefetch(self):
        return self.prefetch_related(
            models.Prefetch(
                "components",
                queryset=SystemComponent.objects.only("id", "system_id", "name"),
            )
        )

    def active(self):
        return self.filter(status="active")


class SystemComponentQuerySet(models.QuerySet):
    def for_system(self, system_id):
        return self.filter(system_id=system_id)


class SensorDataQuerySet(models.QuerySet):
    def recent_for_system(self, system_id, limit=1000):
        return (
            self.filter(system_id=system_id)
            .select_related("component")
            .only("timestamp", "value", "unit", "component_id")
            .order_by("-timestamp")[:limit]
        )

    def for_component_range(self, component_id, ts_from, ts_to):
        return (
            self.filter(component_id=component_id, timestamp__range=(ts_from, ts_to))
            .only("timestamp", "value", "unit")
            .order_by("timestamp")
        )

    def with_system_component(self):
        return self.select_related("system", "component")


class DiagnosticReportQuerySet(models.QuerySet):
    def recent_for_system(self, system_id, limit=100):
        return (
            self.filter(system_id=system_id)
            .only("id", "title", "severity", "status", "created_at", "ai_confidence")
            .order_by("-created_at")[:limit]
        )

    def open_critical(self):
        return self.filter(status="open", severity="critical")


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
    description = models.TextField(blank=True)
    system_type = models.CharField(max_length=50, choices=SYSTEM_TYPES, db_index=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="active", db_index=True
    )
    criticality = models.CharField(max_length=20, default="medium", db_index=True)
    location = models.CharField(max_length=200, blank=True)
    installation_date = models.DateField(null=True, blank=True)

    # Денормализация для быстрых ответов API
    components_count = models.PositiveIntegerField(default=0)
    last_reading_at = models.DateTimeField(null=True, blank=True, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = HydraulicSystemQuerySet.as_manager()

    class Meta:
        db_table = "hydraulic_system"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "system_type"], name="idx_hs_status_type"),
            models.Index(fields=["name"], name="idx_hs_name"),
        ]
        constraints = [
            models.CheckConstraint(check=~Q(name=""), name="chk_hs_name_not_empty"),
        ]

    def clean(self):
        if self.status == "inactive" and self.components_count > 0:
            raise ValidationError("Inactive system cannot have components_count > 0")

    def __str__(self):
        return self.name


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
    specification = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = SystemComponentQuerySet.as_manager()

    class Meta:
        db_table = "system_component"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["system", "name"], name="uniq_component_name_per_system"
            ),
        ]
        indexes = [
            models.Index(fields=["system", "name"], name="idx_comp_system_name"),
            # Для JSON оптимизации используем GIN (добавим через миграцию RunSQL с opclass)
        ]

    def __str__(self):
        return f"{self.system_id}::{self.name}"


class SensorData(models.Model):
    """Высокочастотные IoT данные датчиков."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, db_index=True)
    component = models.ForeignKey(
        SystemComponent, on_delete=models.CASCADE, db_index=True
    )
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    unit = models.CharField(max_length=32)
    value = models.FloatField()

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

    objects = SensorDataQuerySet.as_manager()

    class Meta:
        db_table = "sensor_data"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["system", "-timestamp"], name="idx_sd_system_ts_desc"),
            models.Index(
                fields=["component", "-timestamp"], name="idx_sd_comp_ts_desc"
            ),
            models.Index(
                fields=["system", "component", "timestamp"], name="idx_sd_sys_comp_ts"
            ),
            models.Index(fields=["day_bucket"], name="idx_sd_day_bucket"),
        ]
        constraints = [
            models.CheckConstraint(check=~Q(unit=""), name="chk_sd_unit_not_empty"),
            models.CheckConstraint(
                check=Q(value__isnull=False) | Q(value_decimal__isnull=False),
                name="chk_sd_value_present",
            ),
        ]

    def clean(self):
        if self.value is None and self.value_decimal is None:
            raise ValidationError("Either value or value_decimal must be provided")

    def __str__(self):
        return f"{self.system_id}:{self.component_id}@{self.timestamp}"


class DiagnosticReport(models.Model):
    """Диагностический отчёт."""

    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]
    STATUS_CHOICES = [
        ("open", "Open"),
        ("in_progress", "In Progress"),
        ("closed", "Closed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, db_index=True)
    title = models.CharField(max_length=255, db_index=True)
    severity = models.CharField(max_length=16, choices=SEVERITY_CHOICES, db_index=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, db_index=True)
    ai_confidence = models.DecimalField(
        max_digits=5, decimal_places=2, default=Decimal("0.00")
    )  # 0..100
    impacted_components_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    objects = DiagnosticReportQuerySet.as_manager()

    class Meta:
        db_table = "diagnostic_report"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["system", "-created_at"], name="idx_dr_system_created_desc"
            ),
            models.Index(fields=["severity", "status"], name="idx_dr_severity_status"),
        ]
        constraints = [
            models.CheckConstraint(
                check=Q(ai_confidence__gte=0) & Q(ai_confidence__lte=100),
                name="chk_dr_ai_conf_0_100",
            )
        ]

    def __str__(self):
        return f"{self.title} ({self.severity}/{self.status})"
