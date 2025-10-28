"""Diagnostics models (ordered, typed, explicit export)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar
import uuid

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models.functions import TruncDay
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


# -------------------- HydraulicSystem -------------------- #


class HydraulicSystemQuerySet(models.QuerySet["HydraulicSystem"]):
    def with_owner(self) -> HydraulicSystemQuerySet:
        """Выполняет with owner.

        Returns:
            None: None

        """
        return self.select_related("owner")

    def active(self) -> HydraulicSystemQuerySet:
        """Выполняет active.

        Returns:
            None: None

        """
        return self.filter(status="active")


class HydraulicSystem(models.Model):
    SYSTEM_TYPES: ClassVar[list[tuple[str, str]]] = [
        ("industrial", "Промышленная"),
        ("mobile", "Мобильная"),
        ("marine", "Морская"),
        ("aviation", "Авиационная"),
        ("construction", "Строительная"),
        ("mining", "Горнодобывающая"),
        ("agricultural", "Сельскохозяйственная"),
    ]
    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("active", "Активна"),
        ("maintenance", "На обслуживании"),
        ("inactive", "Неактивна"),
        ("emergency", "Аварийная"),
        ("decommissioned", "Списана"),
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    name: models.CharField = models.CharField(max_length=200, db_index=True)
    description: models.TextField = models.TextField(blank=True, default="")
    system_type: models.CharField = models.CharField(
        max_length=50, choices=SYSTEM_TYPES, db_index=True
    )
    status: models.CharField = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="active", db_index=True
    )

    owner: models.ForeignKey = models.ForeignKey(
        "users.User",
        on_delete=models.PROTECT,
        related_name="hydraulic_systems",
        db_index=True,
    )

    components_count: models.PositiveIntegerField = models.PositiveIntegerField(
        default=0
    )
    last_reading_at: models.DateTimeField = models.DateTimeField(
        null=True, blank=True, db_index=True
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )
    updated_at: models.DateTimeField = models.DateTimeField(
        auto_now=True, db_index=True
    )

    objects = models.Manager()
    qs: HydraulicSystemQuerySet = HydraulicSystemQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        components: RelatedManager[SystemComponent]
        sensor_data: RelatedManager[SensorData]
        diagnostic_reports: RelatedManager[DiagnosticReport]

    class Meta:
        db_table = "diagnostics_hydraulicsystem"
        ordering = ["-updated_at"]
        indexes = [
            BTreeIndex(fields=["owner", "status"], name="idx_hs_owner_status"),
            BrinIndex(
                fields=["updated_at"], autosummarize=True, name="brin_hs_updated"
            ),
            BrinIndex(
                fields=["last_reading_at"],
                autosummarize=True,
                name="brin_hs_last_reading",
            ),
        ]

    def __str__(self) -> str:
        """Возвращает строковое представление объекта.

        Returns:
            None: None

        """
        return f"{self.name} ({self.system_type})"


# -------------------- SystemComponent -------------------- #


class SystemComponentQuerySet(models.QuerySet["SystemComponent"]):
    def for_system(self, system_id: uuid.UUID) -> SystemComponentQuerySet:
        """Выполняет for system.

        Args:
            system_id (int): Идентификатор system

        Returns:
            None: None

        """
        return self.filter(system_id=system_id)


class SystemComponent(models.Model):
    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem,
        related_name="components",
        on_delete=models.CASCADE,
        db_index=True,
    )
    name: models.CharField = models.CharField(max_length=255)

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    qs: SystemComponentQuerySet = SystemComponentQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        sensor_data: RelatedManager[SensorData]

    class Meta:
        db_table = "diagnostics_systemcomponent"
        ordering = ["name"]
        indexes = [BTreeIndex(fields=["system", "name"], name="idx_comp_system_name")]

    def __str__(self) -> str:
        """Возвращает строковое представление объекта.

        Returns:
            None: None

        """
        sys_name = str(getattr(self.system, "name", ""))
        comp_name = str(getattr(self, "name", ""))
        return f"{sys_name}::{comp_name}"


# -------------------- SensorData -------------------- #


class SensorDataQuerySet(models.QuerySet["SensorData"]):
    def for_system(self, system_id: uuid.UUID) -> SensorDataQuerySet:
        """Выполняет for system.

        Args:
            system_id (int): Идентификатор system

        Returns:
            None: None

        """
        return self.filter(system_id=system_id).select_related("component")

    def time_range(self, start: datetime, end: datetime) -> SensorDataQuerySet:
        """Выполняет time range.

        Args:
            start (Any): Параметр start
            end (Any): Параметр end

        Returns:
            None: None

        """
        return self.filter(timestamp__gte=start, timestamp__lt=end)


class SensorData(models.Model):
    SENSOR_TYPES: ClassVar[list[tuple[str, str]]] = [
        ("pressure", "Давление"),
        ("temperature", "Температура"),
        ("flow", "Поток"),
        ("vibration", "Вибрация"),
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="sensor_data",
        db_index=True,
    )
    component: models.ForeignKey = models.ForeignKey(
        SystemComponent,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sensor_data",
        db_index=True,
    )

    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)
    sensor_type: models.CharField = models.CharField(
        max_length=64, choices=SENSOR_TYPES, db_index=True
    )
    value: models.FloatField = models.FloatField(
        validators=[MinValueValidator(float("-inf"))]
    )
    unit: models.CharField = models.CharField(max_length=32, default="", blank=True)

    is_critical: models.BooleanField = models.BooleanField(default=False, db_index=True)
    warning_message: models.CharField = models.CharField(
        max_length=240, default="", blank=True
    )

    day_bucket: models.GeneratedField = models.GeneratedField(
        expression=TruncDay("timestamp"),
        output_field=models.DateField(),
        db_persist=True,
        db_index=True,
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )

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
        """Выполняет clean.

        Returns:
            None: None

        """
        if self.timestamp and self.timestamp > timezone.now() + timedelta(minutes=5):
            raise ValidationError(
                "Timestamp cannot be more than 5 minutes in the future"
            )

    def save(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Сохраняет объект модели в базу данных.

        Returns:
            None: None

        """
        self.full_clean()
        super().save(*args, **kwargs)
        sys_pk = getattr(self.system, "pk", None)
        if sys_pk is not None and self.timestamp:
            HydraulicSystem.objects.filter(id=sys_pk).update(
                last_reading_at=self.timestamp
            )

    def __str__(self) -> str:
        """Возвращает строковое представление объекта.

        Returns:
            None: None

        """
        comp_name = str(getattr(self.component, "name", "N/A"))
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"{self.sensor_type}@{sys_name}:{comp_name}"


# -------------------- DiagnosticReport -------------------- #


class DiagnosticReportQuerySet(models.QuerySet["DiagnosticReport"]):
    def recent_for_system(
        self, system_id: uuid.UUID, limit: int = 100
    ) -> DiagnosticReportQuerySet:
        """Выполняет recent for system.

        Args:
            system_id (int): Идентификатор system
            limit (Any): Параметр limit

        Returns:
            None: None

        """
        return (
            self.filter(system_id=system_id)
            .only("id", "title", "severity", "status", "created_at")
            .order_by("-created_at")[:limit]
        )


class DiagnosticReport(models.Model):
    SEVERITY_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("info", "Информация"),
        ("warning", "Предупреждение"),
        ("error", "Ошибка"),
        ("critical", "Критическая"),
    ]
    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("open", "Открыт"),
        ("in_progress", "В процессе"),
        ("closed", "Закрыт"),
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="diagnostic_reports",
        db_index=True,
    )
    title: models.CharField = models.CharField(max_length=255, db_index=True)
    severity: models.CharField = models.CharField(
        max_length=16, choices=SEVERITY_CHOICES, db_index=True
    )
    status: models.CharField = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default="open", db_index=True
    )

    ai_confidence: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0
    )

    description: models.TextField = models.TextField(blank=True, default="")

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    qs: DiagnosticReportQuerySet = DiagnosticReportQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_diagnosticreport"
        ordering = ["-created_at"]
        indexes = [
            BTreeIndex(fields=["system", "created_at"], name="idx_dr_system_created"),
            BTreeIndex(
                fields=["severity", "created_at"], name="idx_dr_severity_created"
            ),
        ]

    def clean(self) -> None:
        """Выполняет clean.

        Returns:
            None: None

        """
        if not (0.0 <= float(self.ai_confidence) <= 1.0):
            raise ValidationError("AI confidence must be between 0.0 and 1.0")

    def __str__(self) -> str:
        """Возвращает строковое представление объекта.

        Returns:
            None: None

        """
        return f"{self.title} ({self.severity}/{self.status})"


# -------------------- Diagnostic Results (Sprint 1) -------------------- #


class MathematicalModelResultQuerySet(models.QuerySet["MathematicalModelResult"]):
    def for_system(self, system_id: uuid.UUID) -> "MathematicalModelResultQuerySet":
        return self.filter(system_id=system_id)

    def recent(self, limit: int = 100) -> "MathematicalModelResultQuerySet":
        return self.order_by("-timestamp")[:limit]


class MathematicalModelResult(models.Model):
    """Результаты математической модели диагностики.

    Пороговые уровни:
    - Delta < 5% => normal
    - 5% <= Delta < 10% => warning
    - Delta >= 10% => fault
    """

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),
        ("warning", "Предупреждение"),
        ("fault", "Неисправность"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="math_model_results", db_index=True
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    # Измеренные значения
    measured_pressure: models.FloatField = models.FloatField(help_text="Измеренное давление, МПа")
    measured_flow: models.FloatField = models.FloatField(help_text="Измеренный расход, л/мин")
    measured_speed: models.FloatField = models.FloatField(help_text="Измеренная скорость, об/мин")

    # Расчетные значения
    calculated_pressure: models.FloatField = models.FloatField(help_text="Расчетное давление, МПа")
    calculated_flow: models.FloatField = models.FloatField(help_text="Расчетный расход, л/мин")
    calculated_speed: models.FloatField = models.FloatField(help_text="Расчетная скорость, об/мин")

    # Отклонения в процентах
    pressure_deviation: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0)])
    flow_deviation: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0)])
    speed_deviation: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0)])

    max_deviation: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0)], db_index=True, help_text="Максимальное отклонение, %"
    )

    status: models.CharField = models.CharField(max_length=20, choices=STATUS_CHOICES, db_index=True)
    score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], db_index=True, help_text="Оценка 0.0-1.0"
    )

    recommendations: models.TextField = models.TextField(blank=True, default="")
    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)

    objects = models.Manager()
    qs: MathematicalModelResultQuerySet = MathematicalModelResultQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_mathmodelresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_mmr_system_ts"),
            BTreeIndex(fields=["status", "timestamp"], name="idx_mmr_status_ts"),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_mmr_ts"),
        ]

    def __str__(self) -> str:
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"MathModel[{sys_name}]: {self.status} (Delta={self.max_deviation:.1f}%)"


class PhasePortraitResultQuerySet(models.QuerySet["PhasePortraitResult"]):
    def for_system(self, system_id: uuid.UUID) -> "PhasePortraitResultQuerySet":
        return self.filter(system_id=system_id)


class PhasePortraitResult(models.Model):
    """Результаты анализа фазовых портретов.

    Пороговые уровни по DeltaS (%): <10 норма, 10-25 предотказ, >=25 отказ.
    """

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),
        ("pre_fault", "Предотказ"),
        ("fault", "Отказ"),
    ]
    PORTRAIT_TYPES: ClassVar[list[tuple[str, str]]] = [
        ("velocity_position", "V=f(x)"),
        ("force_velocity", "F=f(V)"),
        ("pressure_flow", "P=f(Q)"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="phase_portrait_results", db_index=True
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    recording_duration: models.FloatField = models.FloatField(default=20.0, help_text="Длительность, сек")
    sampling_frequency: models.FloatField = models.FloatField(default=100.0, help_text="Частота, Гц")
    portrait_type: models.CharField = models.CharField(max_length=30, choices=PORTRAIT_TYPES, db_index=True)

    calculated_area: models.FloatField = models.FloatField()
    reference_area: models.FloatField = models.FloatField()
    area_deviation: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0)], db_index=True)

    center_shift_x: models.FloatField = models.FloatField(default=0.0)
    center_shift_y: models.FloatField = models.FloatField(default=0.0)
    contour_breaks: models.IntegerField = models.IntegerField(default=0)
    shape_distortion: models.FloatField = models.FloatField(default=0.0)

    status: models.CharField = models.CharField(max_length=20, choices=STATUS_CHOICES, db_index=True)
    score: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], db_index=True)

    recommendations: models.TextField = models.TextField(blank=True, default="")
    portrait_image_path: models.CharField = models.CharField(max_length=255, blank=True, default="")
    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)

    objects = models.Manager()
    qs: PhasePortraitResultQuerySet = PhasePortraitResultQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_phaseportraitresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_ppr_system_ts"),
            BTreeIndex(fields=["portrait_type", "timestamp"], name="idx_ppr_type_ts"),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_ppr_ts"),
        ]

    def __str__(self) -> str:
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"PhasePortrait[{sys_name}]: {self.portrait_type} (DeltaS={self.area_deviation:.1f}%)"


class TribodiagnosticResultQuerySet(models.QuerySet["TribodiagnosticResult"]):
    def for_system(self, system_id: uuid.UUID) -> "TribodiagnosticResultQuerySet":
        return self.filter(system_id=system_id)


class TribodiagnosticResult(models.Model):
    """Результаты трибодиагностики масел/жидкостей (ISO 4406 и химический анализ)."""

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),
        ("attention", "Внимание"),
        ("critical", "Критическое"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="tribo_results", db_index=True
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    sample_volume: models.FloatField = models.FloatField(default=100.0)
    sample_location: models.CharField = models.CharField(max_length=100, default="Основной бак")

    # ISO 4406
    particles_4um: models.IntegerField = models.IntegerField(default=0)
    particles_6um: models.IntegerField = models.IntegerField(default=0)
    particles_14um: models.IntegerField = models.IntegerField(default=0)
    iso_class: models.CharField = models.CharField(max_length=20, db_index=True, help_text="Напр. 15/13/10")

    # Элементы (ppm)
    iron_ppm: models.FloatField = models.FloatField(default=0.0)
    copper_ppm: models.FloatField = models.FloatField(default=0.0)
    aluminum_ppm: models.FloatField = models.FloatField(default=0.0)
    chromium_ppm: models.FloatField = models.FloatField(default=0.0)
    silicon_ppm: models.FloatField = models.FloatField(default=0.0)

    # Физ.-хим. свойства
    viscosity_cst: models.FloatField = models.FloatField(default=0.0)
    ph_level: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(14.0)], default=7.0
    )
    water_content_ppm: models.FloatField = models.FloatField(default=0.0)

    wear_source: models.CharField = models.CharField(max_length=100, blank=True, default="")

    status: models.CharField = models.CharField(max_length=20, choices=STATUS_CHOICES, db_index=True)
    score: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], db_index=True)

    lab_report_number: models.CharField = models.CharField(max_length=50, blank=True, default="")
    analyzed_by: models.CharField = models.CharField(max_length=100, blank=True, default="")

    recommendations: models.TextField = models.TextField(blank=True, default="")

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    analysis_date: models.DateTimeField = models.DateTimeField(db_index=True)

    objects = models.Manager()
    qs: TribodiagnosticResultQuerySet = TribodiagnosticResultQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_tribodiagnosticresult"
        ordering = ["-analysis_date"]
        indexes = [
            BTreeIndex(fields=["system", "analysis_date"], name="idx_tdr_system_date"),
            BTreeIndex(fields=["iso_class", "analysis_date"], name="idx_tdr_iso_date"),
            BrinIndex(fields=["analysis_date"], autosummarize=True, name="brin_tdr_date"),
        ]

    def __str__(self) -> str:
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"Tribo[{sys_name}]: ISO {self.iso_class} - {self.status}"


class IntegratedDiagnosticResultQuerySet(models.QuerySet["IntegratedDiagnosticResult"]):
    def for_system(self, system_id: uuid.UUID) -> "IntegratedDiagnosticResultQuerySet":
        return self.filter(system_id=system_id)

    def recent_summary(self, limit: int = 50) -> "IntegratedDiagnosticResultQuerySet":
        return (
            self.select_related("system")
            .only("id", "system__name", "timestamp", "integrated_score", "overall_status", "predicted_remaining_life")
            .order_by("-timestamp")[:limit]
        )


class IntegratedDiagnosticResult(models.Model):
    """Интегральная оценка диагностики.

    D = 0.4*M + 0.4*P + 0.2*T
    """

    OVERALL_STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),
        ("warning", "Предупреждение"),
        ("fault", "Неисправность"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE, related_name="integrated_results", db_index=True
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    math_result: models.ForeignKey = models.ForeignKey(
        MathematicalModelResult, on_delete=models.SET_NULL, null=True, blank=True, related_name="integrated_results"
    )
    phase_result: models.ForeignKey = models.ForeignKey(
        PhasePortraitResult, on_delete=models.SET_NULL, null=True, blank=True, related_name="integrated_results"
    )
    tribo_result: models.ForeignKey = models.ForeignKey(
        TribodiagnosticResult, on_delete=models.SET_NULL, null=True, blank=True, related_name="integrated_results"
    )

    math_score: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    phase_score: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    tribo_score: models.FloatField = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])

    integrated_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], db_index=True
    )

    overall_status: models.CharField = models.CharField(max_length=20, choices=OVERALL_STATUS_CHOICES, db_index=True)

    predicted_remaining_life: models.IntegerField = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(0)])
    confidence_level: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=0.0
    )

    recommendations: models.TextField = models.TextField()
    priority_actions: models.JSONField = models.JSONField(default=list, blank=True)

    diagnosis_duration: models.FloatField = models.FloatField(default=0.0)
    data_quality_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], default=1.0
    )

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)

    objects = models.Manager()
    qs: IntegratedDiagnosticResultQuerySet = IntegratedDiagnosticResultQuerySet.as_manager()  # type: ignore[assignment]

    class Meta:
        db_table = "diagnostics_integratedresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_idr_system_ts"),
            BTreeIndex(fields=["overall_status", "timestamp"], name="idx_idr_status_ts"),
            BTreeIndex(fields=["integrated_score", "timestamp"], name="idx_idr_score_ts"),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_idr_ts"),
        ]

    def save(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # Авторасчет интегральной оценки и статуса
        self.integrated_score = 0.4 * self.math_score + 0.4 * self.phase_score + 0.2 * self.tribo_score
        if self.integrated_score < 0.3:
            self.overall_status = "normal"
        elif self.integrated_score < 0.6:
            self.overall_status = "warning"
        else:
            self.overall_status = "fault"
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"Integrated[{sys_name}]: D={self.integrated_score:.2f} ({self.overall_status})"


__all__ = [
    "DiagnosticReport",
    "HydraulicSystem",
    "SensorData",
    "SystemComponent",
    "MathematicalModelResult",
    "PhasePortraitResult",
    "TribodiagnosticResult",
    "IntegratedDiagnosticResult",
]
