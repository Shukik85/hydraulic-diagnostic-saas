from django.db import models
from django.contrib.auth import get_user_model
<<<<<<< HEAD
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import json
import uuid

User = get_user_model()

class HydraulicSystem(models.Model):
    """Модель гидравлической системы"""
    SYSTEM_TYPES = [
        ('industrial', 'Промышленная'),
        ('mobile', 'Мобильная'),
        ('marine', 'Морская'),
        ('aviation', 'Авиационная'),
        ('construction', 'Строительная'),
        ('mining', 'Горнодобывающая'),
        ('agricultural', 'Сельскохозяйственная'),
=======

User = get_user_model()


class HydraulicSystem(models.Model):
    """Гидравлическая система"""
    
    SYSTEM_TYPES = [
        ('industrial', 'Промышленная'),
        ('mobile', 'Мобильная'),  
        ('marine', 'Морская'),
        ('aviation', 'Авиационная'),
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
    ]
    
    STATUS_CHOICES = [
        ('active', 'Активна'),
        ('maintenance', 'На обслуживании'),
        ('inactive', 'Неактивна'),
<<<<<<< HEAD
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
        # Простой алгоритм оценки здоровья на основе критических событий
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
    sensor_type = models.CharField(max_length=50, choices=SENSOR_TYPES, verbose_name='Тип датчика')
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

class MaintenanceRecord(models.Model):
    """Записи о техническом обслуживании"""
    MAINTENANCE_TYPES = [
        ('preventive', 'Профилактическое'),
        ('corrective', 'Корректирующее'),
        ('emergency', 'Экстренное'),
        ('scheduled', 'Плановое'),
        ('condition_based', 'По состоянию'),
    ]
    
    STATUS_CHOICES = [
        ('planned', 'Запланировано'),
        ('in_progress', 'В процессе'),
        ('completed', 'Завершено'),
        ('cancelled', 'Отменено'),
        ('overdue', 'Просрочено'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, verbose_name='Система')
    title = models.CharField(max_length=200, verbose_name='Название работ')
    description = models.TextField(verbose_name='Описание')
    
    # Тип и статус
    maintenance_type = models.CharField(max_length=50, choices=MAINTENANCE_TYPES, verbose_name='Тип обслуживания')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='planned', verbose_name='Статус')
    
    # Временные рамки
    scheduled_date = models.DateTimeField(verbose_name='Запланированная дата')
    started_at = models.DateTimeField(null=True, blank=True, verbose_name='Начато')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='Завершено')
    estimated_duration = models.DurationField(null=True, blank=True, verbose_name='Ожидаемая продолжительность')
    
    # Ответственные
    assigned_to = models.ForeignKey(
        User, on_delete=models.SET_NULL, 
        null=True, blank=True,
        related_name='assigned_maintenance',
        verbose_name='Назначено'
    )
    completed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, 
        null=True, blank=True,
        related_name='completed_maintenance',
        verbose_name='Выполнено'
    )
    
    # Детали работ
    work_performed = models.TextField(blank=True, verbose_name='Выполненные работы')
    parts_replaced = models.JSONField(default=list, verbose_name='Замененные детали')
    materials_used = models.JSONField(default=list, verbose_name='Использованные материалы')
    
    # Стоимость
    estimated_cost = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        verbose_name='Ожидаемая стоимость'
    )
    actual_cost = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        verbose_name='Фактическая стоимость'
    )
    
    # Результаты
    success = models.BooleanField(null=True, blank=True, verbose_name='Успешно выполнено')
    notes = models.TextField(blank=True, verbose_name='Заметки')
    attachments = models.JSONField(default=list, verbose_name='Вложения')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Запись ТО'
        verbose_name_plural = 'Записи ТО'
        ordering = ['-scheduled_date']
        indexes = [
            models.Index(fields=['system', 'status']),
            models.Index(fields=['scheduled_date']),
            models.Index(fields=['assigned_to', 'status']),
        ]
    
    def __str__(self):
        return f"{self.system.name} - {self.title} ({self.scheduled_date.strftime('%d.%m.%Y')})"
    
    def is_overdue(self):
        """Проверить просрочено ли обслуживание"""
        return (self.status == 'planned' and 
                self.scheduled_date < timezone.now())

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
        ('replaced', 'Заменен'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, verbose_name='Система')
    name = models.CharField(max_length=200, verbose_name='Название')
    component_type = models.CharField(max_length=50, choices=COMPONENT_TYPES, verbose_name='Тип компонента')
    
    # Технические характеристики
    manufacturer = models.CharField(max_length=100, blank=True, verbose_name='Производитель')
    model = models.CharField(max_length=100, blank=True, verbose_name='Модель')
    serial_number = models.CharField(max_length=100, blank=True, verbose_name='Серийный номер')
    part_number = models.CharField(max_length=100, blank=True, verbose_name='Номер детали')
    
    # Статус и состояние
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='operational', verbose_name='Статус')
    condition_score = models.FloatField(
        default=100.0,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        verbose_name='Оценка состояния (%)'
    )
    
    # Даты
    installation_date = models.DateField(null=True, blank=True, verbose_name='Дата установки')
    last_inspection = models.DateTimeField(null=True, blank=True, verbose_name='Последняя проверка')
    next_maintenance = models.DateTimeField(null=True, blank=True, verbose_name='Следующее ТО')
    
    # Дополнительные данные
    specifications = models.JSONField(default=dict, blank=True, verbose_name='Технические характеристики')
    maintenance_history = models.JSONField(default=list, blank=True, verbose_name='История обслуживания')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Компонент системы'
        verbose_name_plural = 'Компоненты системы'
        ordering = ['system', 'component_type', 'name']
        indexes = [
            models.Index(fields=['system', 'component_type']),
            models.Index(fields=['system', 'status']),
        ]
    
    def __str__(self):
        return f"{self.system.name} - {self.name} ({self.get_component_type_display()})"

class Alert(models.Model):
    """Системные оповещения"""
    ALERT_TYPES = [
        ('sensor_critical', 'Критические показания датчика'),
        ('maintenance_due', 'Требуется ТО'),
        ('maintenance_overdue', 'Просрочено ТО'),
        ('system_failure', 'Отказ системы'),
        ('anomaly_detected', 'Обнаружена аномалия'),
        ('communication_lost', 'Потеря связи'),
        ('data_quality', 'Проблемы с качеством данных'),
        ('performance_degradation', 'Снижение производительности'),
    ]
    
=======
        ('faulty', 'Неисправна'),
    ]
    
    name = models.CharField(max_length=200, verbose_name='Название системы')
    system_type = models.CharField(max_length=50, choices=SYSTEM_TYPES, verbose_name='Тип системы')
    location = models.CharField(max_length=200, verbose_name='Местоположение')
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='active', verbose_name='Статус')
    
    # Технические характеристики
    max_pressure = models.FloatField(verbose_name='Максимальное давление (бар)')
    flow_rate = models.FloatField(verbose_name='Расход (л/мин)')
    temperature_range = models.CharField(max_length=50, verbose_name='Диапазон температур')
    
    # Метаданные
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='Владелец')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Дата обновления')
    
    class Meta:
        db_table = 'hydraulic_systems'
        verbose_name = 'Гидравлическая система'
        verbose_name_plural = 'Гидравлические системы'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.get_system_type_display()})"


class SensorData(models.Model):
    """Данные с датчиков"""
    
    SENSOR_TYPES = [
        ('pressure', 'Датчик давления'),
        ('temperature', 'Датчик температуры'),
        ('flow', 'Датчик расхода'),
        ('vibration', 'Датчик вибрации'),
        ('level', 'Датчик уровня'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='sensor_data', verbose_name='Система')
    sensor_type = models.CharField(max_length=50, choices=SENSOR_TYPES, verbose_name='Тип датчика')
    value = models.FloatField(verbose_name='Значение')
    unit = models.CharField(max_length=20, verbose_name='Единица измерения')
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='Время измерения')
    
    class Meta:
        db_table = 'sensor_data'
        verbose_name = 'Данные датчика'
        verbose_name_plural = 'Данные датчиков'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.get_sensor_type_display()}: {self.value} {self.unit}"


class Equipment(models.Model):
    """Оборудование гидравлической системы"""
    
    EQUIPMENT_TYPES = [
        ('pump', 'Насос'),
        ('valve', 'Клапан'),
        ('cylinder', 'Цилиндр'),
        ('motor', 'Мотор'),
        ('filter', 'Фильтр'),
        ('accumulator', 'Аккумулятор'),
    ]
    
    STATUS_CHOICES = [
        ('operational', 'В работе'),
        ('maintenance', 'На обслуживании'),
        ('faulty', 'Неисправно'),
        ('retired', 'Списано'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='equipment', verbose_name='Система')
    name = models.CharField(max_length=200, verbose_name='Название')
    equipment_type = models.CharField(max_length=50, choices=EQUIPMENT_TYPES, verbose_name='Тип оборудования')
    manufacturer = models.CharField(max_length=200, verbose_name='Производитель')
    model = models.CharField(max_length=200, verbose_name='Модель')
    serial_number = models.CharField(max_length=100, unique=True, verbose_name='Серийный номер')
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='operational', verbose_name='Статус')
    
    # Даты обслуживания
    installation_date = models.DateField(verbose_name='Дата установки')
    last_maintenance = models.DateField(null=True, blank=True, verbose_name='Последнее обслуживание')
    next_maintenance = models.DateField(null=True, blank=True, verbose_name='Следующее обслуживание')
    
    # Метаданные
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Дата обновления')
    
    class Meta:
        db_table = 'equipment'
        verbose_name = 'Оборудование'
        verbose_name_plural = 'Оборудование'
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.get_equipment_type_display()})"


class DiagnosticReport(models.Model):
    """Отчет о диагностике"""
    
    REPORT_TYPES = [
        ('routine', 'Плановая проверка'),
        ('emergency', 'Экстренная диагностика'),
        ('post_repair', 'После ремонта'),
        ('commissioning', 'Ввод в эксплуатацию'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='reports', verbose_name='Система')
    report_type = models.CharField(max_length=50, choices=REPORT_TYPES, verbose_name='Тип отчета')
    summary = models.TextField(verbose_name='Сводка')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='reports_created', verbose_name='Создан')
    
    class Meta:
        db_table = 'diagnostic_reports'
        verbose_name = 'Отчет о диагностике'
        verbose_name_plural = 'Отчеты о диагностике'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Отчет {self.id} - {self.get_report_type_display()} ({self.created_at.date()})"


class Diagnosis(models.Model):
    """Результаты диагностики"""
    
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
    SEVERITY_LEVELS = [
        ('low', 'Низкая'),
        ('medium', 'Средняя'),
        ('high', 'Высокая'),
        ('critical', 'Критическая'),
    ]
    
    STATUS_CHOICES = [
<<<<<<< HEAD
        ('active', 'Активно'),
        ('acknowledged', 'Принято'),
        ('resolved', 'Решено'),
        ('dismissed', 'Отклонено'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, verbose_name='Система')
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPES, verbose_name='Тип оповещения')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, verbose_name='Уровень серьезности')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name='Статус')
    
    # Содержание оповещения
    title = models.CharField(max_length=200, verbose_name='Заголовок')
    message = models.TextField(verbose_name='Сообщение')
    
    # Связанные данные
    related_sensor_data = models.ForeignKey(
        SensorData, on_delete=models.CASCADE, 
        null=True, blank=True,
        verbose_name='Связанные данные датчика'
    )
    metadata = models.JSONField(default=dict, blank=True, verbose_name='Дополнительные данные')
    
    # Обработка
    acknowledged_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, 
        null=True, blank=True,
        related_name='acknowledged_alerts',
        verbose_name='Принято пользователем'
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True, verbose_name='Время принятия')
    resolved_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, 
        null=True, blank=True,
        related_name='resolved_alerts',
        verbose_name='Решено пользователем'
    )
    resolved_at = models.DateTimeField(null=True, blank=True, verbose_name='Время решения')
    resolution_notes = models.TextField(blank=True, verbose_name='Заметки по решению')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создано')
    
    class Meta:
        verbose_name = 'Оповещение'
        verbose_name_plural = 'Оповещения'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['system', 'status', '-created_at']),
            models.Index(fields=['severity', 'status', '-created_at']),
            models.Index(fields=['alert_type', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.system.name} - {self.title} ({self.get_severity_display()})"
    
    def acknowledge(self, user):
        """Принять оповещение"""
        self.status = 'acknowledged'
        self.acknowledged_by = user
        self.acknowledged_at = timezone.now()
        self.save()
    
    def resolve(self, user, notes=""):
        """Решить оповещение"""
        self.status = 'resolved'
        self.resolved_by = user
        self.resolved_at = timezone.now()
        self.resolution_notes = notes
        self.save()
=======
        ('pending', 'В ожидании'),
        ('in_progress', 'В процессе'),
        ('completed', 'Завершена'),
        ('failed', 'Неудачная'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='diagnoses', verbose_name='Система')
    equipment = models.ForeignKey(Equipment, on_delete=models.SET_NULL, null=True, blank=True, related_name='diagnoses', verbose_name='Оборудование')
    report = models.ForeignKey(DiagnosticReport, on_delete=models.CASCADE, related_name='diagnoses', verbose_name='Отчет')
    
    # Основная информация
    title = models.CharField(max_length=200, verbose_name='Заголовок')
    description = models.TextField(verbose_name='Описание')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, verbose_name='Серьезность')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')
    
    # Результаты
    findings = models.TextField(blank=True, verbose_name='Результаты')
    recommendations = models.TextField(blank=True, verbose_name='Рекомендации')
    
    # Даты
    diagnosed_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата диагностики')
    resolved_at = models.DateTimeField(null=True, blank=True, verbose_name='Дата решения')
    
    # Ответственные
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='diagnoses_created', verbose_name='Создано')
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='diagnoses_assigned', verbose_name='Назначено')
    
    class Meta:
        db_table = 'diagnoses'
        verbose_name = 'Диагностика'
        verbose_name_plural = 'Диагностики'
        ordering = ['-diagnosed_at']
    
    def __str__(self):
        return f"{self.title} - {self.get_severity_display()} ({self.diagnosed_at.date()})"
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
