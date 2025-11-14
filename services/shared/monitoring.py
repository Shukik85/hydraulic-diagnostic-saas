# services/shared/monitoring.py
"""
Shared monitoring utilities для всех microservices.
Prometheus metrics, health checks, status tracking.
"""
import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)


# === Prometheus Metrics ===

# Request metrics
REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['service', 'method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['service', 'method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Service health metrics
SERVICE_UP = Gauge(
    'service_up',
    'Service is up and running',
    ['service']
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['service', 'database']
)

# Resource metrics
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage', ['service'])
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes', ['service'])
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['service', 'gpu_id'])

# ML-specific metrics
INFERENCE_COUNT = Counter(
    'ml_inference_total',
    'Total ML inferences',
    ['service', 'model', 'status']
)

INFERENCE_LATENCY = Histogram(
    'ml_inference_duration_seconds',
    'ML inference latency',
    ['service', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)


class HealthChecker:
    """
    Health checker для microservice dependencies.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.checks = {}
    
    async def check_database(self, db_connection) -> bool:
        """Проверка подключения к БД."""
        try:
            await db_connection.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def check_redis(self, redis_client) -> bool:
        """Проверка Redis."""
        try:
            await redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def check_disk_space(self, threshold_percent: float = 90.0) -> bool:
        """Проверка свободного места."""
        try:
            usage = psutil.disk_usage('/')
            if usage.percent > threshold_percent:
                logger.warning(f"Low disk space: {usage.percent}%")
                return False
            return True
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False
    
    def check_memory(self, threshold_percent: float = 90.0) -> bool:
        """Проверка памяти."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > threshold_percent:
                logger.warning(f"High memory usage: {memory.percent}%")
                return False
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    async def get_health_status(self, db=None, redis=None) -> Dict[str, Any]:
        """Comprehensive health check."""
        checks = {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        # Database check
        if db:
            db_healthy = await self.check_database(db)
            checks["checks"]["database"] = "ok" if db_healthy else "failed"
            if not db_healthy:
                checks["status"] = "unhealthy"
        
        # Redis check
        if redis:
            redis_healthy = await self.check_redis(redis)
            checks["checks"]["redis"] = "ok" if redis_healthy else "failed"
            if not redis_healthy:
                checks["status"] = "degraded"
        
        # System resources
        disk_ok = self.check_disk_space()
        memory_ok = self.check_memory()
        
        checks["checks"]["disk"] = "ok" if disk_ok else "warning"
        checks["checks"]["memory"] = "ok" if memory_ok else "warning"
        
        if not (disk_ok and memory_ok) and checks["status"] == "healthy":
            checks["status"] = "degraded"
        
        return checks


class MetricsCollector:
    """
    Сбор метрик для Prometheus.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        SERVICE_UP.labels(service=service_name).set(1)
    
    def update_resource_metrics(self):
        """Обновление метрик ресурсов."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        CPU_USAGE.labels(service=self.service_name).set(cpu_percent)
        
        # Memory
        memory = psutil.virtual_memory()
        MEMORY_USAGE.labels(service=self.service_name).set(memory.used)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Запись HTTP запроса."""
        REQUESTS_TOTAL.labels(
            service=self.service_name,
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        REQUEST_DURATION.labels(
            service=self.service_name,
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_inference(self, model: str, status: str, duration: float):
        """Запись ML inference."""
        INFERENCE_COUNT.labels(
            service=self.service_name,
            model=model,
            status=status
        ).inc()
        
        INFERENCE_LATENCY.labels(
            service=self.service_name,
            model=model
        ).observe(duration)


def generate_metrics() -> bytes:
    """Generate Prometheus metrics."""
    return generate_latest()
