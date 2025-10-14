from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class HydraulicSystem(models.Model):
    """Гидравлическая система"""
    
    SYSTEM_TYPES = [
        ('industrial', 'Промышленная'),
        ('mobile', 'Мобильная'),  
        ('marine', 'Морская'),
        ('aviation', 'Авиационная'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Активна'),
        ('maintenance', 'На обслуживании'),
        ('inactive', 'Неактивна'),
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
    
    # Критические значения
    is_critical = models.BooleanField(default=False, verbose_name='Критическое значение')
    threshold_exceeded = models.BooleanField(default=False, verbose_name='Превышен порог')
    
    class Meta:
        db_table = 'sensor_data'
        verbose_name = 'Данные датчика'
        verbose_name_plural = 'Данные датчиков'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.get_sensor_type_display()} - {self.value}{self.unit} ({self.timestamp})"


class Equipment(models.Model):
    """Оборудование гидросистемы"""
    
    EQUIPMENT_TYPES = [
        ('pump', 'Насос'),
        ('valve', 'Клапан'),
        ('cylinder', 'Цилиндр'),
        ('motor', 'Гидромотор'),
        ('filter', 'Фильтр'),
        ('accumulator', 'Аккумулятор'),
    ]
    
    STATUS_CHOICES = [
        ('operational', 'Рабочее'),
        ('maintenance', 'На обслуживании'),
        ('faulty', 'Неисправное'),
        ('retired', 'Списано'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='equipment', verbose_name='Система')
    name = models.CharField(max_length=200, verbose_name='Название')
    equipment_type = models.CharField(max_length=50, choices=EQUIPMENT_TYPES, verbose_name='Тип оборудования')
    manufacturer = models.CharField(max_length=200, verbose_name='Производитель')
    model = models.CharField(max_length=100, verbose_name='Модель')
    serial_number = models.CharField(max_length=100, unique=True, verbose_name='Серийный номер')
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='operational', verbose_name='Статус')
    
    # Даты
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


class Diagnosis(models.Model):
    """Результаты диагностики"""
    
    SEVERITY_LEVELS = [
        ('low', 'Низкая'),
        ('medium', 'Средняя'),
        ('high', 'Высокая'),
        ('critical', 'Критическая'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'В ожидании'),
        ('in_progress', 'В процессе'),
        ('completed', 'Завершена'),
        ('failed', 'Неудачная'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='diagnoses', verbose_name='Система')
    equipment = models.ForeignKey(Equipment, on_delete=models.SET_NULL, null=True, blank=True, related_name='diagnoses', verbose_name='Оборудование')
    
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
