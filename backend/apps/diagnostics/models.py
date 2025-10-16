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
    
    # Статус и критичность
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name='Статус')
    criticality = models.CharField(max_length=20, choices=CRITICALITY_LEVELS, default='medium', verbose_name='Критичность')
    
    # Местоположение
    location = models.CharField(max_length=200, blank=True, verbose_name='Местоположение')
    installation_date = models.DateField(null=True, blank=True, verbose_name='Дата установки')
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создан')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлен')
    last_maintenance = models.DateTimeField(null=True, blank=True, verbose_name='Последнее ТО')
    next_maintenance = models.DateTimeField(null=True, blank=True, verbose_name='Следующее ТО')
    
    class Meta:
        verbose_name = 'Гидравлическая система'
        verbose_name_plural = 'Гидравлические системы'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'criticality']),
            models.Index(fields=['system_type']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_system_type_display()})"
    
    def is_operational(self):
        """Проверка, что система в рабочем состоянии"""
        return self.status == 'active'
    
    def needs_maintenance(self):
        """Проверка необходимости технического обслуживания"""
        if self.next_maintenance:
            return self.next_maintenance <= timezone.now()
        return False


class DiagnosticReport(models.Model):
    """Отчет диагностики гидравлической системы"""
    
    SEVERITY_CHOICES = [
        ('info', 'Информация'),
        ('warning', 'Предупреждение'),
        ('critical', 'Критично'),
        ('emergency', 'Аварийно'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'В ожидании'),
        ('in_progress', 'В процессе'),
        ('completed', 'Завершен'),
        ('archived', 'Архивирован'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE,
        related_name='diagnostic_reports',
        verbose_name='Система'
    )
    
    # Основная информация
    title = models.CharField(max_length=200, verbose_name='Заголовок')
    description = models.TextField(verbose_name='Описание')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, verbose_name='Важность')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='Статус')
    
    # Технические данные
    measured_pressure = models.FloatField(null=True, blank=True, verbose_name='Измеренное давление')
    measured_flow = models.FloatField(null=True, blank=True, verbose_name='Измеренный расход')
    measured_temperature = models.FloatField(null=True, blank=True, verbose_name='Измеренная температура')
    vibration_level = models.FloatField(null=True, blank=True, verbose_name='Уровень вибрации')
    noise_level = models.FloatField(null=True, blank=True, verbose_name='Уровень шума')
    
    # Рекомендации
    recommendations = models.TextField(blank=True, verbose_name='Рекомендации')
    actions_taken = models.TextField(blank=True, verbose_name='Выполненные действия')
    
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
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(
        HydraulicSystem, on_delete=models.CASCADE,
        related_name='components',
        verbose_name='Система'
    )
    
    component_type = models.CharField(max_length=50, choices=COMPONENT_TYPES, verbose_name='Тип компонента')
    name = models.CharField(max_length=200, verbose_name='Название')
    manufacturer = models.CharField(max_length=100, blank=True, verbose_name='Производитель')
    model = models.CharField(max_length=100, blank=True, verbose_name='Модель')
    serial_number = models.CharField(max_length=100, blank=True, verbose_name='Серийный номер')
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='operational', verbose_name='Статус')
    installation_date = models.DateField(null=True, blank=True, verbose_name='Дата установки')
    last_service_date = models.DateField(null=True, blank=True, verbose_name='Дата последнего обслуживания')
    
    notes = models.TextField(blank=True, verbose_name='Примечания')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Создан')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Обновлен')
    
    class Meta:
        verbose_name = 'Компонент системы'
        verbose_name_plural = 'Компоненты системы'
        ordering = ['system', 'component_type', 'name']
        indexes = [
            models.Index(fields=['system', 'component_type']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.system.name} - {self.get_component_type_display()}: {self.name}"
