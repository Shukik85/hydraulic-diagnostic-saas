"""
Health Check Service for ML Inference
Provides readiness/liveness checks for CPU, memory, models and cache.
"""

import time

import psutil
import structlog

from config import HEALTH_METRICS
from models.ensemble import EnsembleModel

from .cache_service import CacheService

logger = structlog.get_logger()


class HealthCheckService:
    """Enterprise health checks for ML service."""

    def __init__(self) -> None:
        self.started_at = time.time()

    async def check_health(
        self,
        ensemble: EnsembleModel | None = None,
        cache: CacheService | None = None,
    ) -> dict[str, object]:
        """Return structured health status with resource and dependency checks."""
        process = psutil.Process()
        vm = psutil.virtual_memory()

        cpu_percent = process.cpu_percent()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        total_mb = vm.total / (1024 * 1024)

        cpu_ok = cpu_percent < HEALTH_METRICS.get("cpu_threshold", 80.0)
        mem_ok = (rss_mb / max(total_mb, 1.0)) * 100.0 < HEALTH_METRICS.get("memory_threshold", 85.0)

        models_ok = bool(ensemble and ensemble.is_ready())
        cache_ok = bool(cache and await cache.is_connected())

        healthy = cpu_ok and mem_ok and models_ok and cache_ok

        status = {
            "healthy": healthy,
            "status": "ok" if healthy else "degraded",
            "checks": {
                "cpu_ok": cpu_ok,
                "memory_ok": mem_ok,
                "models_ok": models_ok,
                "cache_ok": cache_ok,
            },
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_usage_mb": rss_mb,
                "memory_total_mb": total_mb,
                "uptime_seconds": time.time() - self.started_at,
            },
            "uptime_seconds": time.time() - self.started_at,
        }

        logger.debug("Health check computed", **status)
        return status
