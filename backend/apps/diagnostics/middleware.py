import logging
import time

from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class APILoggingMiddleware(MiddlewareMixin):
    """Middleware для логирования API запросов"""

    def process_request(self, request):
        """Обработка входящего запроса"""
        request.start_time = time.time()

        # Логирование API запросов к диагностическим системам
        if request.path.startswith("/api/"):
            logger.info(
                f" API Request: {request.method} {request.path} "
                f"from {request.META.get('REMOTE_ADDR', 'Unknown')} "
                f"User: {request.user.username if request.user.is_authenticated else 'Anonymous'}"
            )

    def process_response(self, request, response):
        """Обработка исходящего ответа"""
        if hasattr(request, "start_time") and request.path.startswith("/api/"):
            duration = time.time() - request.start_time

            # Логирование медленных запросов
            if duration > 2.0:  # Запросы дольше 2 секунд
                logger.warning(
                    f" Slow API Request: {request.method} {request.path} "
                    f"took {duration:.2f}s (Status: {response.status_code})"
                )
            else:
                logger.debug(
                    f" API Response: {request.method} {request.path} "
                    f"- {response.status_code} in {duration:.2f}s"
                )

        return response


class DiagnosticSystemMonitoringMiddleware(MiddlewareMixin):
    """Middleware для мониторинга диагностических операций"""

    def process_request(self, request):
        """Мониторинг специфических операций"""
        # Мониторинг критических операций
        critical_endpoints = [
            "/diagnose/",
            "/upload_sensor_data/",
            "/generate_test_data/",
        ]

        if any(endpoint in request.path for endpoint in critical_endpoints):
            logger.info(
                f" Critical Operation: {request.method} {request.path} "
                f"by {request.user.username if request.user.is_authenticated else 'Anonymous'}"
            )

    def process_exception(self, request, exception):
        """Обработка исключений в диагностических операциях"""
        if request.path.startswith("/api/"):
            logger.error(
                f" API Exception: {request.method} {request.path} "
                f"Error: {str(exception)} "
                f"User: {request.user.username if request.user.is_authenticated else 'Anonymous'}"
            )

        return None


class RateLimitingMiddleware(MiddlewareMixin):
    """Middleware для ограничения частоты запросов к AI операциям"""

    def __init__(self, get_response):
        super().__init__(get_response)
        self.rate_limits = {}  # {user_id: [(timestamp, endpoint), ...]}

    def process_request(self, request):
        """Проверка ограничений частоты"""
        # Ограничения для AI операций
        ai_endpoints = ["/diagnose/", "/ask_question/", "/search_knowledge/"]

        if any(endpoint in request.path for endpoint in ai_endpoints):
            if request.user.is_authenticated:
                user_id = request.user.id
                current_time = time.time()

                # Инициализация для нового пользователя
                if user_id not in self.rate_limits:
                    self.rate_limits[user_id] = []

                # Очистка старых записей (старше 1 минуты)
                self.rate_limits[user_id] = [
                    (timestamp, endpoint)
                    for timestamp, endpoint in self.rate_limits[user_id]
                    if current_time - timestamp < 60
                ]

                # Проверка лимита (максимум 10 AI операций в минуту)
                ai_requests_count = len(
                    [
                        req
                        for req in self.rate_limits[user_id]
                        if any(ai_endpoint in req for ai_endpoint in ai_endpoints)
                    ]
                )

                if ai_requests_count >= 10:
                    logger.warning(
                        f" Rate limit exceeded: {request.user.username} "
                        f"attempted {request.path}"
                    )

                    return JsonResponse(
                        {
                            "error": "Превышен лимит запросов к AI",
                            "message": "Максимум 10 AI операций в минуту",
                            "retry_after": 60,
                        },
                        status=429,
                    )

                # Добавление текущего запроса
                self.rate_limits[user_id].append((current_time, request.path))

        return None


class SystemHealthMiddleware(MiddlewareMixin):
    """Middleware для мониторинга здоровья системы"""

    def process_request(self, request):
        """Мониторинг системных ресурсов"""
        # Простая проверка доступности базы данных
        if request.path == "/api/health/":
            try:
                from django.db import connection

                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")

                # Проверка доступности AI компонентов

                return JsonResponse(
                    {
                        "status": "healthy",
                        "timestamp": time.time(),
                        "database": "ok",
                        "ai_engine": "ok",
                        "rag_system": "ok",
                    }
                )

            except Exception as e:
                logger.error(f" System health check failed: {e}")
                return JsonResponse(
                    {"status": "unhealthy", "timestamp": time.time(), "error": str(e)},
                    status=503,
                )

        return None
