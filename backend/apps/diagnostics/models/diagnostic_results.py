"""Diagnostic results models for mathematical model, phase portraits and tribodiagnostics."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

from django.contrib.postgres.indexes import BTreeIndex, BrinIndex
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


# -------------------- Mathematical Model Results -------------------- #


class MathematicalModelResultQuerySet(models.QuerySet["MathematicalModelResult"]):
    """QuerySet для результатов математической модели."""

    def for_system(
        self, system_id: uuid.UUID
    ) -> MathematicalModelResultQuerySet:
        """Фильтр по системе.

        Args:
            system_id: UUID системы

        Returns:
            Отфильтрованный QuerySet
        """
        return self.filter(system_id=system_id)

    def recent(self, limit: int = 100) -> MathematicalModelResultQuerySet:
        """Последние результаты.

        Args:
            limit: Количество записей

        Returns:
            QuerySet с последними результатами
        """
        return self.order_by("-timestamp")[:limit]


class MathematicalModelResult(models.Model):
    """Результат диагностики математической модели.

    Диагностирует отклонения фактических параметров от расчетных
    по формулам гидравлики.
    """

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),  # Delta < 5%
        ("warning", "Предупреждение"),  # 5% <= Delta < 10%
        ("fault", "Неисправность"),  # Delta >= 10%
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        "diagnostics.HydraulicSystem",
        on_delete=models.CASCADE,
        related_name="math_model_results",
        db_index=True,
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    # Измеренные значения
    measured_pressure: models.FloatField = models.FloatField(
        help_text="Измеренное давление, МПа"
    )
    measured_flow: models.FloatField = models.FloatField(
        help_text="Измеренный расход, л/мин"
    )
    measured_speed: models.FloatField = models.FloatField(
        help_text="Измеренная скорость, об/мин"
    )

    # Расчетные значения
    calculated_pressure: models.FloatField = models.FloatField(
        help_text="Расчетное давление, МПа"
    )
    calculated_flow: models.FloatField = models.FloatField(
        help_text="Расчетный расход, л/мин"
    )
    calculated_speed: models.FloatField = models.FloatField(
        help_text="Расчетная скорость, об/мин"
    )

    # Отклонения в %
    pressure_deviation: models.FloatField = models.FloatField(
        help_text="Отклонение давления, %",
        validators=[MinValueValidator(0.0)],
    )
    flow_deviation: models.FloatField = models.FloatField(
        help_text="Отклонение расхода, %",
        validators=[MinValueValidator(0.0)],
    )
    speed_deviation: models.FloatField = models.FloatField(
        help_text="Отклонение скорости, %",
        validators=[MinValueValidator(0.0)],
    )

    # Общее отклонение (максимальное)
    max_deviation: models.FloatField = models.FloatField(
        help_text="Максимальное отклонение, %",
        validators=[MinValueValidator(0.0)],
        db_index=True,
    )

    status: models.CharField = models.CharField(
        max_length=20, choices=STATUS_CHOICES, db_index=True
    )
    score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка от 0.0 до 1.0 (вес 40% в интегральной оценке)",
        db_index=True,
    )

    # Рекомендации
    recommendations: models.TextField = models.TextField(
        blank=True, default="", help_text="Рекомендации по результатам анализа"
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )

    objects = models.Manager()
    qs: MathematicalModelResultQuerySet = (
        MathematicalModelResultQuerySet.as_manager()  # type: ignore[assignment]
    )

    class Meta:
        db_table = "diagnostics_mathmodelresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_mmr_system_ts"),
            BTreeIndex(fields=["status", "timestamp"], name="idx_mmr_status_ts"),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_mmr_ts"),
        ]

    def __str__(self) -> str:
        """Строковое представление.

        Returns:
            Строковое представление объекта
        """
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"MathModel[{sys_name}]: {self.status} (Delta={self.max_deviation:.1f}%)"


# -------------------- Phase Portrait Results -------------------- #


class PhasePortraitResultQuerySet(models.QuerySet["PhasePortraitResult"]):
    """QuerySet для результатов фазовых портретов."""

    def for_system(
        self, system_id: uuid.UUID
    ) -> PhasePortraitResultQuerySet:
        """Фильтр по системе.

        Args:
            system_id: UUID системы

        Returns:
            Отфильтрованный QuerySet
        """
        return self.filter(system_id=system_id)


class PhasePortraitResult(models.Model):
    """Результат анализа фазовых портретов.

    Анализирует площадь и форму фазовых портретов V=f(x), F=f(V), P=f(Q)
    для выявления предотказных состояний.
    """

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),  # DeltaS < 10%
        ("pre_fault", "Предотказ"),  # 10% <= DeltaS < 25%
        ("fault", "Отказ"),  # DeltaS >= 25%
    ]

    PORTRAIT_TYPES: ClassVar[list[tuple[str, str]]] = [
        ("velocity_position", "V=f(x) - Скорость от положения"),
        ("force_velocity", "F=f(V) - Усилие от скорости"),
        ("pressure_flow", "P=f(Q) - Давление от расхода"),
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        "diagnostics.HydraulicSystem",
        on_delete=models.CASCADE,
        related_name="phase_portrait_results",
        db_index=True,
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    # Параметры записи
    recording_duration: models.FloatField = models.FloatField(
        help_text="Длительность записи, сек", default=20.0
    )
    sampling_frequency: models.FloatField = models.FloatField(
        help_text="Частота дискретизации, Гц", default=100.0
    )
    portrait_type: models.CharField = models.CharField(
        max_length=30, choices=PORTRAIT_TYPES, db_index=True
    )

    # Результаты анализа площади
    calculated_area: models.FloatField = models.FloatField(
        help_text="Вычисленная площадь фазового портрета"
    )
    reference_area: models.FloatField = models.FloatField(
        help_text="Эталонная площадь для сравнения"
    )
    area_deviation: models.FloatField = models.FloatField(
        help_text="Отклонение площади, %",
        validators=[MinValueValidator(0.0)],
        db_index=True,
    )

    # Анализ формы портрета
    center_shift_x: models.FloatField = models.FloatField(
        help_text="Смещение центра портрета по X", default=0.0
    )
    center_shift_y: models.FloatField = models.FloatField(
        help_text="Смещение центра портрета по Y", default=0.0
    )
    contour_breaks: models.IntegerField = models.IntegerField(
        help_text="Количество разрывов контура", default=0
    )
    shape_distortion: models.FloatField = models.FloatField(
        help_text="Коэффициент искажения формы", default=0.0
    )

    status: models.CharField = models.CharField(
        max_length=20, choices=STATUS_CHOICES, db_index=True
    )
    score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка от 0.0 до 1.0 (вес 40% в интегральной оценке)",
        db_index=True,
    )

    # Рекомендации
    recommendations: models.TextField = models.TextField(
        blank=True, default="", help_text="Рекомендации по анализу портрета"
    )

    # Путь к файлу с изображением портрета (опционально)
    portrait_image_path: models.CharField = models.CharField(
        max_length=255, blank=True, default="", help_text="Путь к изображению портрета"
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )

    objects = models.Manager()
    qs: PhasePortraitResultQuerySet = (
        PhasePortraitResultQuerySet.as_manager()  # type: ignore[assignment]
    )

    class Meta:
        db_table = "diagnostics_phaseportraitresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_ppr_system_ts"),
            BTreeIndex(
                fields=["portrait_type", "timestamp"], name="idx_ppr_type_ts"
            ),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_ppr_ts"),
        ]

    def __str__(self) -> str:
        """Строковое представление.

        Returns:
            Строковое представление объекта
        """
        sys_name = str(getattr(self.system, "name", "N/A"))
        return (
            f"PhasePortrait[{sys_name}]: {self.portrait_type} - "
            f"{self.status} (DeltaS={self.area_deviation:.1f}%)"
        )


# -------------------- Tribodiagnostic Results -------------------- #


class TribodiagnosticResultQuerySet(models.QuerySet["TribodiagnosticResult"]):
    """QuerySet для результатов трибодиагностики."""

    def for_system(
        self, system_id: uuid.UUID
    ) -> TribodiagnosticResultQuerySet:
        """Фильтр по системе.

        Args:
            system_id: UUID системы

        Returns:
            Отфильтрованный QuerySet
        """
        return self.filter(system_id=system_id)


class TribodiagnosticResult(models.Model):
    """Результат трибодиагностики (анализ проб жидкости).

    Анализирует чистоту жидкости, элементный состав и физико-химические
    свойства для определения источников износа.
    """

    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),  # ISO 15/13/10
        ("attention", "Внимание"),  # ISO 16/14/11
        ("critical", "Критическое"),  # ISO >=17/15/12
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        "diagnostics.HydraulicSystem",
        on_delete=models.CASCADE,
        related_name="tribo_results",
        db_index=True,
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    # Параметры пробы
    sample_volume: models.FloatField = models.FloatField(
        help_text="Объем пробы, мл", default=100.0
    )
    sample_location: models.CharField = models.CharField(
        max_length=100,
        help_text="Место отбора пробы",
        default="Основной бак",
    )

    # ISO 4406 - чистота по частицам
    particles_4um: models.IntegerField = models.IntegerField(
        help_text="Частицы >=4 мкм на мл", default=0
    )
    particles_6um: models.IntegerField = models.IntegerField(
        help_text="Частицы >=6 мкм на мл", default=0
    )
    particles_14um: models.IntegerField = models.IntegerField(
        help_text="Частицы >=14 мкм на мл", default=0
    )
    iso_class: models.CharField = models.CharField(
        max_length=20,
        help_text="Класс чистоты ISO 4406 (например, 15/13/10)",
        db_index=True,
    )

    # Элементный анализ (ppm)
    iron_ppm: models.FloatField = models.FloatField(
        help_text="Содержание железа, ppm", default=0.0
    )
    copper_ppm: models.FloatField = models.FloatField(
        help_text="Содержание меди, ppm", default=0.0
    )
    aluminum_ppm: models.FloatField = models.FloatField(
        help_text="Содержание алюминия, ppm", default=0.0
    )
    chromium_ppm: models.FloatField = models.FloatField(
        help_text="Содержание хрома, ppm", default=0.0
    )
    silicon_ppm: models.FloatField = models.FloatField(
        help_text="Содержание кремния, ppm", default=0.0
    )

    # Физико-химические свойства
    viscosity_cst: models.FloatField = models.FloatField(
        help_text="Вязкость, сСт", default=0.0
    )
    ph_level: models.FloatField = models.FloatField(
        help_text="Уровень pH",
        validators=[MinValueValidator(0.0), MaxValueValidator(14.0)],
        default=7.0,
    )
    water_content_ppm: models.FloatField = models.FloatField(
        help_text="Содержание воды, ppm", default=0.0
    )

    # Источники износа (автоматическое определение)
    wear_source: models.CharField = models.CharField(
        max_length=100,
        help_text="Определенный источник износа",
        blank=True,
        default="",
    )

    status: models.CharField = models.CharField(
        max_length=20, choices=STATUS_CHOICES, db_index=True
    )
    score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка от 0.0 до 1.0 (вес 20% в интегральной оценке)",
        db_index=True,
    )

    # Лабораторная информация
    lab_report_number: models.CharField = models.CharField(
        max_length=50, blank=True, default="", help_text="Номер отчета лаборатории"
    )
    analyzed_by: models.CharField = models.CharField(
        max_length=100, blank=True, default="", help_text="Кем проанализировано"
    )

    # Рекомендации
    recommendations: models.TextField = models.TextField(
        blank=True, default="", help_text="Рекомендации по результатам анализа"
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )
    analysis_date: models.DateTimeField = models.DateTimeField(
        help_text="Дата проведения анализа", db_index=True
    )

    objects = models.Manager()
    qs: TribodiagnosticResultQuerySet = (
        TribodiagnosticResultQuerySet.as_manager()  # type: ignore[assignment]
    )

    class Meta:
        db_table = "diagnostics_tribodiagnosticresult"
        ordering = ["-analysis_date"]
        indexes = [
            BTreeIndex(fields=["system", "analysis_date"], name="idx_tdr_system_date"),
            BTreeIndex(fields=["iso_class", "analysis_date"], name="idx_tdr_iso_date"),
            BrinIndex(
                fields=["analysis_date"], autosummarize=True, name="brin_tdr_date"
            ),
        ]

    def __str__(self) -> str:
        """Строковое представление.

        Returns:
            Строковое представление объекта
        """
        sys_name = str(getattr(self.system, "name", "N/A"))
        return f"Tribo[{sys_name}]: ISO {self.iso_class} - {self.status}"


# -------------------- Integrated Diagnostic Results -------------------- #


class IntegratedDiagnosticResultQuerySet(
    models.QuerySet["IntegratedDiagnosticResult"]
):
    """QuerySet для интегрированных результатов диагностики."""

    def for_system(
        self, system_id: uuid.UUID
    ) -> IntegratedDiagnosticResultQuerySet:
        """Фильтр по системе.

        Args:
            system_id: UUID системы

        Returns:
            Отфильтрованный QuerySet
        """
        return self.filter(system_id=system_id)

    def recent_summary(
        self, limit: int = 50
    ) -> IntegratedDiagnosticResultQuerySet:
        """Последние результаты для сводки.

        Args:
            limit: Количество записей

        Returns:
            QuerySet с последними результатами
        """
        return (
            self.select_related("system")
            .only(
                "id",
                "system__name",
                "timestamp",
                "integrated_score",
                "overall_status",
                "predicted_remaining_life",
            )
            .order_by("-timestamp")[:limit]
        )


class IntegratedDiagnosticResult(models.Model):
    """Интегральный результат диагностики.

    Объединяет результаты всех трех методов по формуле:
    D = 0.4 * M + 0.4 * P + 0.2 * T

    где M - математическая модель, P - фазовые портреты, T - трибодиагностика
    """

    OVERALL_STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ("normal", "Норма"),  # D < 0.3
        ("warning", "Предупреждение"),  # 0.3 <= D < 0.6
        ("fault", "Неисправность"),  # D >= 0.6
    ]

    id: models.UUIDField = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    system: models.ForeignKey = models.ForeignKey(
        "diagnostics.HydraulicSystem",
        on_delete=models.CASCADE,
        related_name="integrated_results",
        db_index=True,
    )
    timestamp: models.DateTimeField = models.DateTimeField(db_index=True)

    # Ссылки на исходные результаты
    math_result: models.ForeignKey = models.ForeignKey(
        MathematicalModelResult,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="integrated_results",
    )
    phase_result: models.ForeignKey = models.ForeignKey(
        PhasePortraitResult,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="integrated_results",
    )
    tribo_result: models.ForeignKey = models.ForeignKey(
        TribodiagnosticResult,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="integrated_results",
    )

    # Индивидуальные оценки (0.0-1.0)
    math_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка математической модели (0.0-1.0)",
    )
    phase_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка фазовых портретов (0.0-1.0)",
    )
    tribo_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка трибодиагностики (0.0-1.0)",
    )

    # Интегральная оценка
    integrated_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="D = 0.4*M + 0.4*P + 0.2*T",
        db_index=True,
    )

    overall_status: models.CharField = models.CharField(
        max_length=20, choices=OVERALL_STATUS_CHOICES, db_index=True
    )

    # Прогнозирование
    predicted_remaining_life: models.IntegerField = models.IntegerField(
        help_text="Прогнозируемый остаточный ресурс, часы",
        validators=[MinValueValidator(0)],
        null=True,
        blank=True,
    )
    confidence_level: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Уровень доверия прогноза",
        default=0.0,
    )

    # Объединенные рекомендации
    recommendations: models.TextField = models.TextField(
        help_text="Сводные рекомендации по всем методам"
    )
    priority_actions: models.JSONField = models.JSONField(
        default=list,
        help_text="Приоритетные действия в JSON формате",
        blank=True,
    )

    # Метаданные
    diagnosis_duration: models.FloatField = models.FloatField(
        help_text="Время выполнения диагностики, сек", default=0.0
    )
    data_quality_score: models.FloatField = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Оценка качества исходных данных",
        default=1.0,
    )

    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, db_index=True
    )

    objects = models.Manager()
    qs: IntegratedDiagnosticResultQuerySet = (
        IntegratedDiagnosticResultQuerySet.as_manager()  # type: ignore[assignment]
    )

    class Meta:
        db_table = "diagnostics_integratedresult"
        ordering = ["-timestamp"]
        indexes = [
            BTreeIndex(fields=["system", "timestamp"], name="idx_idr_system_ts"),
            BTreeIndex(
                fields=["overall_status", "timestamp"], name="idx_idr_status_ts"
            ),
            BTreeIndex(
                fields=["integrated_score", "timestamp"], name="idx_idr_score_ts"
            ),
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_idr_ts"),
        ]

    def save(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Переопределение save для автоматического расчета интегральной оценки.

        Returns:
            None
        """
        # Автоматический расчет интегральной оценки
        self.integrated_score = (
            0.4 * self.math_score + 0.4 * self.phase_score + 0.2 * self.tribo_score
        )

        # Определение общего статуса
        if self.integrated_score < 0.3:
            self.overall_status = "normal"
        elif self.integrated_score < 0.6:
            self.overall_status = "warning"
        else:
            self.overall_status = "fault"

        super().save(*args, **kwargs)

    def __str__(self) -> str:
        """Строковое представление.

        Returns:
            Строковое представление объекта
        """
        sys_name = str(getattr(self.system, "name", "N/A"))
        return (
            f"Integrated[{sys_name}]: D={self.integrated_score:.2f} "
            f"({self.overall_status})"
        )


__all__ = [
    "MathematicalModelResult",
    "PhasePortraitResult",
    "TribodiagnosticResult",
    "IntegratedDiagnosticResult",
]
