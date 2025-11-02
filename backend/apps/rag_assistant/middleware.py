"""Модуль проекта с автогенерированным докстрингом."""

# apps/rag_assistant/middleware.py
# PERFORMANCE MONITORING MIDDLEWARE

import logging
import threading
import time

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
import structlog

# Логгеры для мониторинга
performance_logger = logging.getLogger("performance")
struct_logger = structlog.get_logger()

# Thread-local стораж для метрик
local_data = threading.local()


class PerformanceMonitoringMiddleware(MiddlewareMixin):
    """Middleware для мониторинга производительности API
    Отслеживает время ответа, меморию, медленные запросы.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.slow_request_threshold = getattr(
            settings, "SLOW_REQUEST_THRESHOLD", 1.0
        )  # 1 секунда
        self.very_slow_threshold = getattr(
            settings, "VERY_SLOW_THRESHOLD", 5.0
        )  # 5 секунд

    def __call__(self, request):
        # Начало замера
        start_time = time.time()
        local_data.start_time = start_time
        local_data.request = request

        # Выполнение запроса
        response = self.get_response(request)

        # Конец замера
        end_time = time.time()
        duration = end_time - start_time

        # Обработка метрик
        self._process_performance_metrics(request, response, duration)

        # Добавляем headers для мониторинга
        response["X-Response-Time"] = f"{duration:.3f}s"
        response["X-Process-Time"] = f"{int(duration * 1000)}ms"

        # Показываем предупреждение о медленном ответе
        if duration > self.very_slow_threshold:
            response["X-Performance-Warning"] = "very-slow"
        elif duration > self.slow_request_threshold:
            response["X-Performance-Warning"] = "slow"

        return response

    def _process_performance_metrics(self, request, response, duration):
        """Обработка и логирование метрик производительности."""
        path = request.path
        method = request.method
        status_code = response.status_code
        user = getattr(request, "user", None)
        user_id = user.id if user and user.is_authenticated else None

        # Метрики для structured logging
        metrics = {
            "event": "http_request",
            "path": path,
            "method": method,
            "duration_ms": round(duration * 1000, 2),
            "status_code": status_code,
            "user_id": user_id,
            "timestamp": time.time(),
        }

        # Логирование медленных запросов
        if duration > self.very_slow_threshold:
            struct_logger.error(
                "Very slow request detected",
                **metrics,
                threshold_ms=self.very_slow_threshold * 1000,
            )
        elif duration > self.slow_request_threshold:
            struct_logger.warning(
                "Slow request detected",
                **metrics,
                threshold_ms=self.slow_request_threshold * 1000,
            )
        # Обычные запросы логируем только на DEBUG
        elif settings.DEBUG:
            struct_logger.info("HTTP request processed", **metrics)

        # Сохраняем метрики в кеш
        self._store_metrics_in_cache(metrics)

        # Отслеживаем AI операции
        if self._is_ai_request(path):
            self._track_ai_metrics(metrics)

    def _store_metrics_in_cache(self, metrics):
        """Сохранение метрик в Redis для мониторинга."""
        try:
            # Общие метрики производительности
            daily_key = f"performance_metrics:{time.strftime('%Y-%m-%d')}"
            hourly_key = f"performance_metrics:{time.strftime('%Y-%m-%d:%H')}"

            # Увеличиваем счетчики
            cache.set(
                f"{daily_key}:requests",
                cache.get(f"{daily_key}:requests", 0) + 1,
                timeout=86400 * 7,
            )  # 7 дней

            cache.set(
                f"{hourly_key}:requests",
                cache.get(f"{hourly_key}:requests", 0) + 1,
                timeout=3600 * 24,
            )  # 24 часа

            # Сохраняем среднее время ответа
            avg_key = f"{hourly_key}:avg_duration"
            current_avg = cache.get(avg_key, 0)
            request_count = cache.get(f"{hourly_key}:requests", 1)
            new_avg = (
                (current_avg * (request_count - 1)) + metrics["duration_ms"]
            ) / request_count
            cache.set(avg_key, round(new_avg, 2), timeout=3600 * 24)

        except Exception as e:
            # Не ломаем основной процесс из-за метрик
            struct_logger.error("Failed to store metrics", error=str(e))

    def _is_ai_request(self, path):
        """Проверка, является ли запрос AI операцией."""
        ai_paths = [
            "/api/rag_assistant/query/",
            "/api/rag_assistant/documents/",
            "/api/rag_assistant/search/",
            "/api/ai/",
            "/api/diagnostics/analyze/",
        ]
        return any(path.startswith(ai_path) for ai_path in ai_paths)

    def _track_ai_metrics(self, metrics):
        """Отдельное отслеживание AI операций."""
        try:
            ai_key = f"ai_metrics:{time.strftime('%Y-%m-%d:%H')}"

            # Увеличиваем счетчик AI запросов
            cache.set(
                f"{ai_key}:ai_requests",
                cache.get(f"{ai_key}:ai_requests", 0) + 1,
                timeout=3600 * 24,
            )

            # Отслеживаем долгие AI операции
            if metrics["duration_ms"] > 10000:  # > 10 секунд
                cache.set(
                    f"{ai_key}:slow_ai_requests",
                    cache.get(f"{ai_key}:slow_ai_requests", 0) + 1,
                    timeout=3600 * 24,
                )

                struct_logger.warning(
                    "Slow AI operation detected", **metrics, category="ai_performance"
                )

        except Exception as e:
            struct_logger.error("Failed to track AI metrics", error=str(e))


class HealthCheckMiddleware(MiddlewareMixin):
    """Middleware для health checks - быстрые ответы без логирования."""

    def __init__(self, get_response):
        self.get_response = get_response
        self.health_paths = [
            "/health/",
            "/api/health/",
            "/readiness/",
            "/liveness/",
        ]

    def __call__(self, request):
        # Быстрый ответ на health checks
        if request.path in self.health_paths:
            return JsonResponse(
                {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "service": "hydraulic-diagnostic-saas",
                }
            )

        return self.get_response(request)


class TimingMiddleware(MiddlewareMixin):
    """Упрощенный middleware для отслеживания времени ответа
    Легковесный и быстрый.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        response = self.get_response(request)
        duration = time.time() - start_time

        response["X-Response-Time"] = f"{duration:.3f}s"
        return response
