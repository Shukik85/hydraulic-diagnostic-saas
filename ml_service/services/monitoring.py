"""
Prometheus Monitoring for ML Service
Enterprise метрики производительности
"""

import time
from typing import Dict, Any

import psutil
import structlog
from prometheus_client import (
    Counter,
    Histogram, 
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest
)

from config import settings

logger = structlog.get_logger()


class MetricsService:
    """
    Prometheus metrics для ML inference сервиса.
    
    Отслеживает:
    - Latency (p50, p95, p99)
    - Throughput (RPS)
    - Error rates
    - Model accuracy
    - System resources
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.start_time = time.time()
        
        # Метрики предсказаний
        self.predictions_total = Counter(
            'ml_predictions_total',
            'Общее количество предсказаний',
            ['model_name', 'prediction_type'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'ml_inference_duration_seconds',
            'Время inference в секундах',
            ['model_name'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.prediction_errors = Counter(
            'ml_prediction_errors_total',
            'Количество ошибок предсказаний',
            ['model_name', 'error_type'],
            registry=self.registry
        )
        
        # Метрики моделей
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Точность моделей',
            ['model_name'],
            registry=self.registry
        )
        
        self.ensemble_score = Gauge(
            'ml_ensemble_score',
            'Текущий ensemble скор',
            registry=self.registry
        )
        
        # Метрики кеша
        self.cache_hits = Counter(
            'ml_cache_hits_total',
            'Количество попаданий в кеш',
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'ml_cache_misses_total',
            'Количество промахов кеша',
            registry=self.registry
        )
        
        # Системные метрики
        self.memory_usage = Gauge(
            'ml_memory_usage_bytes',
            'Использование памяти в байтах',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'ml_cpu_usage_percent',
            'Загрузка CPU в процентах',
            registry=self.registry
        )
        
        # Метаданные сервиса
        self.service_info = Info(
            'ml_service_info',
            'Метаданные ML сервиса',
            registry=self.registry
        )
        
        # Установка метаданных
        self.service_info.info({
            'version': settings.version,
            'name': settings.app_name,
            'max_inference_time_ms': str(settings.max_inference_time_ms)
        })
    
    def update_system_metrics(self) -> None:
        """Обновление системных метрик."""
        try:
            process = psutil.Process()
            
            # Обновляем метрики
            self.memory_usage.set(process.memory_info().rss)
            self.cpu_usage.set(process.cpu_percent())
            
        except Exception as e:
            logger.warning("Failed to update system metrics", error=str(e))
    
    def record_prediction(self, model_name: str, processing_time: float, score: float) -> None:
        """Запись метрик предсказания."""
        self.predictions_total.labels(
            model_name=model_name,
            prediction_type="anomaly"
        ).inc()
        
        self.inference_duration.labels(model_name=model_name).observe(processing_time)
        
        if model_name == "ensemble":
            self.ensemble_score.set(score)
    
    def record_error(self, model_name: str, error_type: str) -> None:
        """Запись ошибки."""
        self.prediction_errors.labels(
            model_name=model_name,
            error_type=error_type
        ).inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Сводка метрик."""
        uptime = time.time() - self.start_time
        
        return {
            "service_uptime_seconds": uptime,
            "predictions_total": self.predictions_total._value.sum(),
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "cpu_usage_percent": psutil.Process().cpu_percent()
        }


# Глобальные метрики
metrics = MetricsService()


def setup_metrics():
    """Инициализация системы метрик."""
    logger.info("Metrics system initialized", 
                prometheus_enabled=settings.enable_metrics,
                metrics_port=settings.metrics_port)
    
    # Обновляем начальные метрики
    metrics.update_system_metrics()


async def periodic_metrics_update():
    """Периодическое обновление метрик."""
    while True:
        try:
            metrics.update_system_metrics()
            await asyncio.sleep(30)  # Обновление каждые 30 секунд
        except Exception as e:
            logger.warning("Periodic metrics update failed", error=str(e))
            await asyncio.sleep(60)  # Увеличиваем интервал при ошибках