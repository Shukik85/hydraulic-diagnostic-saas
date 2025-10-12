from django.db import models

# Create your models here.
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
    warning_message = models.TextField(blank=True, null=True, verbose_name='Предупреждение')
    
    class Meta:
        db_table = 'sensor_data'
        verbose_name = 'Данные датчика'
        verbose_name_plural = 'Данные датчиков'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.system.name} - {self.get_sensor_type_display()}: {self.value} {self.unit}"


class DiagnosticReport(models.Model):
    """Отчет диагностики"""
    
    SEVERITY_LEVELS = [
        ('info', 'Информация'),
        ('warning', 'Предупреждение'),
        ('error', 'Ошибка'),
        ('critical', 'Критическое'),
    ]
    
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='reports', verbose_name='Система')
    title = models.CharField(max_length=200, verbose_name='Заголовок')
    description = models.TextField(verbose_name='Описание')
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, verbose_name='Уровень серьезности')
    
    # RAG-ассистент данные
    ai_analysis = models.TextField(blank=True, null=True, verbose_name='Анализ ИИ')
    recommendations = models.TextField(blank=True, null=True, verbose_name='Рекомендации')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата создания')
    resolved_at = models.DateTimeField(blank=True, null=True, verbose_name='Дата решения')
    
    class Meta:
        db_table = 'diagnostic_reports'
        verbose_name = 'Отчет диагностики'
        verbose_name_plural = 'Отчеты диагностики'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.system.name} - {self.title}"
