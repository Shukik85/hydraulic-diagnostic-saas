"""Diagnostics models (ordered, typed, explicit export)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Q
from django.db.models.functions import TruncDay
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


# -------------------- HydraulicSystem -------------------- #


class HydraulicSystemQuerySet(models.QuerySet["HydraulicSystem"]):
    def with_owner(self) -> "HydraulicSystemQuerySet":
        return self.select_related("owner")

    def active(self) -> "HydraulicSystemQuerySet":
        return self.filter(status="active")


class HydraulicSystem(models.Model):
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

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name: models.CharField = models.CharField(max_length=200, db_index=True)
    description: models.TextField = models.TextField(blank=True, default="")
    system_type: models.CharField = models.CharField(max_length=50, choices=SYSTEM_TYPES, db_index=True)
    status: models.CharField = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active", db_index=True)

    owner: models.ForeignKey = models.ForeignKey(
        "users.User", on_delete=models.PROTECT, related_name="hydraulic_systems", db_index=True
    )

    components_count: models.PositiveIntegerField = models.PositiveIntegerField(default=0)
    last_reading_at: models.DateTimeField = models.DateTimeField(null=True, blank=True, db_index=True)

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True, db_index=True)

    objects = models.Manager()
    qs: HydraulicSystemQuerySet = HydraulicSystemQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        components: RelatedManager["SystemComponent"]
        sensor_data: RelatedManager["SensorData"]
        diagnostic_reports: RelatedManager["DiagnosticReport"]

    class Meta:
        db_table = "diagnostics_hydraulicsystem"
        ordering = ["-updated_at"]
        indexes = [
            BTreeIndex(fields=["owner", "status"], name="idx_hs_owner_status"),
            BrinIndex(fields=["updated_at"], autosummarize=True, name="brin_hs_updated"),
            BrinIndex(fields=["last_reading_at"], autosummarize=True, name="brin_hs_last_reading"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.system_type})"


# -------------------- SystemComponent -------------------- #


class SystemComponentQuerySet(models.QuerySet["SystemComponent"]):
    def for_system(self, system_id: uuid.UUID) -> "SystemComponentQuerySet":
        return self.filter(system_id=system_id)


class SystemComponent(models.Model):
    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, related_name="components", on_delete=models.CASCADE, db_index=True
    )
    name: models.CharField = models.CharField(max_length=255)

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    qs: SystemComponentQuerySet = SystemComponentQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        sensor_data: RelatedManager["SensorData"]

    class Meta:
        db_table = "diagnostics_systemcomponent"
        ordering = ["name"]
        indexes = [BTreeIndex(fields=["system", "name"], name="idx_comp_system_name")]

    def __str__(self) -> str:
        sys_name = str(getattr(self.system, "name", ""))
        comp_name = str(getattr(self, "name", ""))
        return f"{sys_name}::{comp_name}"


# -------------------- SensorData -------------------- #


class SensorDataQuerySet(models.QuerySet["SensorData"]):
    def for_system(self, system_id: uuid.UUID) -> "SensorDataQuerySet":
        return self.filter(system_id=system_id).select_related("component")

    def time_range(self, start: datetime, end: datetime) -> "SensorDataQuerySet":
        return self.filter(timestamp__gte=start, timestamp__lt=end)


class SensorData(models.Model):
    SENSOR_TYPES = [
        ("pressure", "Давление"),
        ("temperature", "Температура"),
        ("flow", "Поток"),
        ("vibration", "Вибрация"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="sensor_data", db_index=True
    )
    component: models.ForeignKey = models.ForeignKey(
        SystemComponent, on_delete=models.SET_NULL, null=True, blank=True, related_name="sensor_data", db_index=True
    )

    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)
    sensor_type: models.CharField = models.CharField(max_length=64, choices=SENSOR_TYPES, db_index=True)
    value: models.FloatField = models.FloatField(validators=[MinValueValidator(float("-inf"))])
    unit: models.CharField = models.CharField(max_length=32, default="", blank=True)

    is_critical: models.BooleanField = models.BooleanField(default=False, db_index=True)
    warning_message: models.CharField = models.CharField(max_length=240, default="", blank=True)

    day_bucket: models.GeneratedField = models.GeneratedField(
        expression=TruncDay("timestamp"), output_field=models.DateField(), db_persist=True, db_index=True
    )

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)

    objects = models.Manager()
    qs: SensorDataQuerySet = SensorDataQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_sensordata"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_sd_system_ts"),
            BTreeIndex(fields=["sensor_type", "timestamp"], name="idx_sd_type_ts"),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_sd_ts"),
        ]

    def clean(self) -> None:
        if self.timestamp and self.timestamp > timezone.now() + timedelta(minutes=5):
            raise ValidationError("Timestamp cannot be more than 5 minutes in the future")

    def save(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.full_clean()
        super().save(*args, **kwargs)
        sys_pk = getattr(self.system, "pk", None)
        if sys_pk is not None and self.timestamp:
            HydraulicSystem.objects.filter(id=sys_pk).update(last_reading_at=self.timestamp)

    def __str__(self) -> str:
        comp_name = str(getattr(self.component, "name", "N/A"))
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"{self.sensor_type}@{sys_name}:{comp_name}"


# -------------------- DiagnosticReport -------------------- #


class DiagnosticReportQuerySet(models.QuerySet["DiagnosticReport"]):
    def recent_for_system(self, system_id: uuid.UUID, limit: int = 100) -> "DiagnosticReportQuerySet":
        return self.filter(system_id=system_id).only("id", "title", "severity", "status", "created_at").order_by("-created_at")[:limit]


class DiagnosticReport(models.Model):
    SEVERITY_CHOICES = [("info", "Информация"), ("warning", "Предупреждение"), ("error", "Ошибка"), ("critical", "Критическая")]
    STATUS_CHOICES = [("open", "Открыт"), ("in_progress", "В процессе"), ("closed", "Закрыт")]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="diagnostic_reports", db_index=True
    )
    title: models.CharField = models.CharField(max_length=255, db_index=True)
    severity: models.CharField = models.CharField(max_length=16, choices=SEVERITY_CHOICES, db_index=True)
    status: models.CharField = models.CharField(max_length=16, choices=STATUS_CHOICES, default="open", db_index=True)

    ai_confidence: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0)

    description: models.TextField = models.TextField(blank=True, default="")

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    qs: DiagnosticReportQuerySet = DiagnosticReportQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_diagnosticreport"
        ordering = ["-created_at"]
        indexes = [
            BTreeIndex(fields=["system", "created_at"], name="idx_dr_system_created"),
            BTreeIndex(fields=["severity", "created_at"], name="idx_dr_severity_created"),
        ]

    def clean(self) -> None:
        if not (0.0 <= float(self.ai_confidence) <= 1.0):
            raise ValidationError("AI confidence must be between 0.0 and 1.0")

    def __str__(self) -> str:
        return f"{self.title} ({self.severity}/{self.status})"


__all__ = [
    "HydraulicSystem",
    "SystemComponent",
    "SensorData",
    "DiagnosticReport",
]
