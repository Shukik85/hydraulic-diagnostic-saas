from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

User = get_user_model()


class HydraulicSystem(models.Model):
    """Гидравлическая система"""

    SYSTEM_TYPES = [
        ('industrial', 'Промышленная'),
        ('mobile', 'Мобильная'),
        ('marine', 'Морская'),
        ('aviation', 'Авиационная'),
        ('construction', 'Строительная'),
        ('mining', 'Горнодобывающая'),
        ('agricultural', 'Сельскохозяйственная'),
    ]

    STATUS_CHOICES = [
        ('active', 'Активна'),
        ('maintenance', 'На обслуживании'),
        ('inactive', 'Неактивна'),
        ('emergency', 'Аварийная'),
        ('decommissioned', 'Списана'),
    ]

    CRITICALITY_LEVELS = [
        ('low', 'Низкая'),
        ('medium', 'Средняя'),
        ('high', 'Высокая'),
        ('critical', 'Критическая'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name='Название системы')
    description = models.TextField(blank=True, verbose_name='Описание')

    # Основные характеристики
    system_type = models.CharField(max_length=50, choices=SYSTEM_TYPES, verbose_name='Тип системы')
    manufacturer = models.CharField(max_length=100, blank=True, verbose_name='Производитель')
    model = models.CharField(max_length=100, blank=True, verbose_name='Модель')
    serial_number = models.CharField(max_length=100, blank=True, verbose_name='Серийный номер')

    # Технические параметры
    max_pressure = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text='Максимальное давление (бар)',
        verbose_name='Макс. давление'
    )
    max_flow = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text='Максимальный расход (л/мин)',
        verbose_name='Макс. расход'
    )
    operating_temperature_min = models.FloatField(
        default=-20, help_text='Мин. рабочая температура (°C)',
        verbose_name='Мин. температура'
    )
    operating_temperature_max = models.FloatField(
        default=80, help_text='Макс. рабочая температура (°C)',
        verbose_name='Макс. температура'
    )
    fluid_type = models.CharField(
        max_length=100, default='Hydraulic Oil ISO 46',
        verbose_name='Тип жидкости'
    )

    # Статус и эксплуатация
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='active', verbose_name='Статус')
    criticality = models.CharField(max_length=50, choices=CRITICALITY_LEVELS, default='medium', verbose_name='Критичность')
    location = models.CharField(max_length=200, blank=True, verbose_name='Местоположение')

    # Даты
    installation_date = models.DateField(null=True, blank=True, verbose_name='Дата установки')
    last_maintenance = models.DateTimeField(null=True, blank=True, verbose_name='Последнее ТО')
    next_maintenance = models.DateTimeField(null=True, blank=True, verbose_name='Следующее ТО')

    # Владелец и метаданные
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='Владелец')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создано')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлено')

    # Дополнительные настройки (JSON)
    custom_parameters = models.JSONField(default=dict, blank=True, verbose_name='Дополнительные параметры')

    class Meta:
        verbose_name = 'Гидравлическая система'
        verbose_name_plural = 'Гидравлические системы'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner', 'status']),
            models.Index(fields=['system_type', 'status']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_system_type_display()})"

    def get_latest_sensor_data(self):
        """Получить последние данные всех датчиков"""
        latest_data = {}
        for sensor_type in ['pressure', 'temperature', 'flow', 'vibration']:
            data = self.sensordata_set.filter(sensor_type=sensor_type).order_by('-timestamp').first()
            if data:
                latest_data[sensor_type] = {
                    'value': data.value,
                    'unit': data.unit,
                    'timestamp': data.timestamp,
                    'is_critical': data.is_critical
                }
        return latest_data

    def get_health_score(self):
        """Рассчитать общий индекс здоровья системы"""
        from datetime import timedelta

        recent_data = self.sensordata_set.filter(
            timestamp__gte=timezone.now() - timedelta(hours=24)
        )

        total_readings = recent_data.count()
        if total_readings == 0:
            return 0

        critical_readings = recent_data.filter(is_critical=True).count()
        health_score = max(0, 100 - (critical_readings / total_readings * 100))

        return round(health_score, 1)


class SensorType(models.Model):
    """Типы датчиков"""
    name = models.CharField(max_length=100, unique=True, verbose_name='Название')
    code = models.CharField(max_length=50, unique=True, verbose_name='Код')
    unit = models.CharField(max_length=20, verbose_name='Единица измерения')
    description = models.TextField(blank=True, verbose_name='Описание')

    # Пороговые значения
    normal_min = models.FloatField(null=True, blank=True, verbose_name='Мин. норма')
    normal_max = models.FloatField(null=True, blank=True, verbose_name='Макс. норма')
    warning_min = models.FloatField(null=True, blank=True, verbose_name='Мин. предупреждение')
    warning_max = models.FloatField(null=True, blank=True, verbose_name='Макс. предупреждение')
    critical_min = models.FloatField(null=True, blank=True, verbose_name='Мин. критичное')
    critical_max = models.FloatField(null=True, blank=True, verbose_name='Макс. критичное')

    is_active = models.BooleanField(default=True, verbose_name='Активен')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Тип датчика'
        verbose_name_plural = 'Типы датчиков'
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.unit})"


class SensorData(models.Model):
    """Данные датчиков"""
    SENSOR_TYPES = [
        ('pressure', 'Давление'),
        ('temperature', 'Температура'),
        ('flow', 'Расход'),
        ('vibration', 'Вибрация'),
        ('contamination', 'Загрязнение'),
        ('noise', 'Шум'),
        ('power', 'Мощность'),
    ]

    STATUS_CHOICES = [
        ('normal', 'Норма'),
        ('warning', 'Предупреждение'),
        ('critical', 'Критично'),
        ('fault', 'Неисправность'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, verbose_name='Система')
    sensor_type = models.CharField(maxlength=50, choices=SENSOR_TYPES, verbose_name='Тип датчика')
    sensor_id = models.CharField(max_length=100, blank=True, verbose_name='ID датчика')

    # Основные данные
    value = models.FloatField(verbose_name='Значение')
    unit = models.CharField(max_length=20, verbose_name='Единица измерения')
    timestamp = models.DateTimeField(default=timezone.now, verbose_name='Время измерения')

    # Статус и оценка
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='normal', verbose_name='Статус')
    is_critical = models.BooleanField(default=False, verbose_name='Критично')
    warning_message = models.TextField(blank=True, verbose_name='Сообщение предупреждения')

    # Дополнительные данные
    raw_data = models.JSONField(default=dict, blank=True, verbose_name='Сырые данные')
    quality_score = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        verbose_name='Качество данных'
    )

    # Вычисляемые поля
    deviation_from_normal = models.FloatField(null=True, blank=True, verbose_name='Отклонение от нормы')
    trend_direction = models.CharField(
        max_length=20,
        choices=[('up', 'Растет'), ('down', 'Падает'), ('stable', 'Стабильно')],
        null=True, blank=True,
        verbose_name='Направление тренда'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Данные датчика'
        verbose_name_plural = 'Данные датчиков'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['system', 'sensor_type', '-timestamp']),
            models.Index(fields=['system', '-timestamp']),
            models.Index(fields=['is_critical', '-timestamp']),
            models.Index(fields=['-timestamp']),
        ]

    def __str__(self):
        return f"{self.system.name} - {self.get_sensor_type_display()}: {self.value} {self.unit}"

    def save(self, *args, **kwargs):
        # Автоматическое определение статуса на основе пороговых значений
        self.determine_status()
        super().save(*args, **kwargs)

    def determine_status(self):
        """Определить статус на основе пороговых значений"""
        try:
            sensor_type = SensorType.objects.get(code=self.sensor_type)

            # Проверка критических значений
            if (sensor_type.critical_min and self.value < sensor_type.critical_min) or \
               (sensor_type.critical_max and self.value > sensor_type.critical_max):
                self.status = 'critical'
                self.is_critical = True
                self.warning_message = f"Критическое значение {self.sensor_type}: {self.value} {self.unit}"

            # Проверка предупреждающих значений
            elif (sensor_type.warning_min and self.value < sensor_type.warning_min) or \
                 (sensor_type.warning_max and self.value > sensor_type.warning_max):
                self.status = 'warning'
                self.is_critical = False
                self.warning_message = f"Предупреждение {self.sensor_type}: {self.value} {self.unit}"

            # Нормальные значения
            else:
                self.status = 'normal'
                self.is_critical = False
                self.warning_message = ''

            # Расчет отклонения от нормы
            if sensor_type.normal_min and sensor_type.normal_max:
                normal_center = (sensor_type.normal_min + sensor_type.normal_max) / 2
                normal_range = sensor_type.normal_max - sensor_type.normal_min
                if normal_range > 0:
                    self.deviation_from_normal = (self.value - normal_center) / normal_range

        except SensorType.DoesNotExist:
            # Если тип датчика не найден, используем базовые правила
            pass


class DiagnosticReport(models.Model):
    """Отчеты диагностики"""
    SEVERITY_LEVELS = [
        ('info', 'Информация'),
        ('warning', 'Предупреждение'),
        ('error', 'Ошибка'),
        ('critical', 'Критично'),
    ]

    REPORT_TYPES = [
        ('automated', 'Автоматический'),
        ('manual', 'Ручной'),
        ('scheduled', 'Плановый'),
        ('emergency', 'Экстренный'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Ожидание'),
        ('processing', 'Обработка'),
        ('completed', 'Завершен'),
        ('failed', 'Ошибка'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, verbose_name='Система')
    title = models.CharField(max_length=200, verbose_name='Заголовок')
    description = models.TextField(verbose_name='Описание')

    # Тип и статус отчета
    report_type = models.CharField(max_length=50, choices=REPORT_TYPES, default='automated', verbose_name='Тип отчета')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='info', verbose_name='Уровень серьезности')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')

    # Результаты диагностики
    findings = models.JSONField(default=list, verbose_name='Результаты')
    recommendations = models.JSONField(default=list, verbose_name='Рекомендации')
    analysis_data = models.JSONField(default=dict, verbose_name='Данные анализа')

    # AI анализ
    ai_confidence = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        verbose_name='Уверенность AI'
    )
    ai_analysis = models.TextField(blank=True, verbose_name='AI анализ')

    # Файлы и вложения
    attachments = models.JSONField(default=list, verbose_name='Вложения')

    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создан')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='Завершен')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлено')

    # Автор (если ручной отчет)
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True,
        verbose_name='Создан пользователем'
    )

    class Meta:
        verbose_name = 'Отчет диагностики'
        verbose_name_plural = 'Отчеты диагностики'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['system', '-created_at']),
            models.Index(fields=['severity', '-created_at']),
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f"{self.system.name} - {self.title} ({self.created_at.strftime('%d.%m.%Y %H:%M')})"

    def mark_completed(self):
        """Отметить отчет как завершенный"""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()


class SystemComponent(models.Model):
    """Компоненты гидравлической системы"""
    COMPONENT_TYPES = [
        ('pump', 'Насос'),
        ('motor', 'Гидромотор'),
        ('cylinder', 'Цилиндр'),
        ('valve', 'Клапан'),
        ('filter', 'Фильтр'),
        ('accumulator', 'Аккумулятор'),
        ('cooler', 'Охладитель'),
        ('tank', 'Бак'),
        ('hose', 'Шланг'),
        ('fitting', 'Фитинг'),
    ]

    STATUS_CHOICES = [
        ('operational', 'Исправен'),
        ('warning', 'Требует внимания'),
        ('maintenance', 'Требует ТО'),
        ('failed', 'Неисправен'),
