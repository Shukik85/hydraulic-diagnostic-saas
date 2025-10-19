import uuid

from django.contrib.auth import get_user_model
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

User = get_user_model()

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
CRITICALITY_LEVELS = [
    ("low", "Низкая"),
    ("medium", "Средняя"),
    ("high", "Высокая"),
    ("critical", "Критическая"),
]


class HydraulicSystem(models.Model):
    """Гидравлическая система"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name="Название системы")
    description = models.TextField(blank=True, verbose_name="Описание")
    system_type = models.CharField(
        max_length=50, choices=SYSTEM_TYPES, verbose_name="Тип системы", db_index=True
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="active",
        verbose_name="Статус",
        db_index=True,
    )
    criticality = models.CharField(
        max_length=20,
        choices=CRITICALITY_LEVELS,
        default="medium",
        verbose_name="Критичность",
        db_index=True,
    )
    location = models.CharField(
        max_length=200, blank=True, verbose_name="Местоположение"
    )
    installation_date = models.DateField(
        null=True, blank=True, verbose_name="Дата установки"
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Гидравлическая система"
        verbose_name_plural = "Гидравлические системы"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["system_type"]),
            models.Index(fields=["status"]),
            models.Index(fields=["criticality"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_system_type_display()})"

    def is_operational(self):
        return self.status == "active"

    def needs_maintenance(self):
        if hasattr(self, "next_maintenance") and self.next_maintenance:
            return self.next_maintenance <= timezone.now()
        return False


class SystemComponent(models.Model):
    """Компонент гидравлической системы"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="components",
        verbose_name="Система",
        db_index=True,
    )
    name = models.CharField(max_length=200, verbose_name="Название компонента")
    specification = models.JSONField(
        blank=True, null=True, verbose_name="Характеристики"
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Компонент системы"
        verbose_name_plural = "Компоненты систем"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["system"]),
        ]

    def __str__(self):
        return f"{self.system.name} – {self.name}"


class SensorData(models.Model):
    """Запись данных с датчика"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="sensor_data",
        verbose_name="Система",
        db_index=True,
    )
    component = models.ForeignKey(
        SystemComponent,
        on_delete=models.CASCADE,
        related_name="sensor_data",
        verbose_name="Компонент",
        db_index=True,
    )
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    value = models.FloatField(verbose_name="Значение")
    unit = models.CharField(max_length=50, blank=True, verbose_name="Единица измерения")

    class Meta:
        verbose_name = "Данные датчика"
        verbose_name_plural = "Данные датчиков"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["system", "timestamp"]),
            models.Index(fields=["component", "timestamp"]),
        ]

    def __str__(self):
        return (
            f"{self.system.name} – {self.component.name} @ {self.timestamp.isoformat()}"
        )


class DiagnosticReport(models.Model):
    """Отчет диагностики"""

    SEVERITY_CHOICES = [
        ("info", "Информация"),
        ("warning", "Предупреждение"),
        ("critical", "Критично"),
        ("emergency", "Аварийно"),
    ]
    STATUS_CHOICES = [
        ("pending", "В ожидании"),
        ("in_progress", "В процессе"),
        ("completed", "Завершен"),
        ("archived", "Архивирован"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem,
        on_delete=models.CASCADE,
        related_name="diagnostic_reports",
        verbose_name="Система",
        db_index=True,
    )
    title = models.CharField(max_length=200, verbose_name="Заголовок")
    severity = models.CharField(
        max_length=20, choices=SEVERITY_CHOICES, verbose_name="Важность", db_index=True
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
        verbose_name="Статус",
        db_index=True,
    )
    ai_confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        verbose_name="Уверенность AI",
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="Создан пользователем",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Отчет диагностики"
        verbose_name_plural = "Отчеты диагностики"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["system", "-created_at"]),
            models.Index(fields=["severity"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.system.name} – {self.title}"

    def mark_completed(self):
        self.status = "completed"
        self.completed_at = timezone.now()
        self.save()
