# apps/core/middleware.py
import logging
from collections.abc import Callable

from asgiref.sync import iscoroutinefunction, markcoroutinefunction
from django.core.cache import cache
from django.http import JsonResponse

logger = logging.getLogger(__name__)


class AsyncRequestLoggingMiddleware:
    """Async middleware - работает только с ASGI."""  # noqa: RUF002

    def __init__(self, get_response):
        self.get_response = get_response
        # Пометить как async если get_response async
        if iscoroutinefunction(self.get_response):
            markcoroutinefunction(self)

    async def __call__(self, request):
        # Код до обработки запроса
        logger.info(f"Async Request: {request.method} {request.path}")

        # Обработка запроса
        response = await self.get_response(request)

        # Код после обработки
        logger.info(f"Async Response: {request.method} {request.path} - {response.status_code}")

        return response


class RateLimitMiddleware:
    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            key = f"rate_limit_{request.user.id}"
            limit = 1000  # requests per hour
        else:
            key = f"rate_limit_{self.get_client_ip(request)}"
            limit = 100

        count = cache.get(key, 0)
        if count >= limit:
            return JsonResponse({"error": "Rate limit exceeded"}, status=429)

        cache.set(key, count + 1, 3600)  # 1 hour
        return self.get_response(request)

    @staticmethod
    def get_client_ip(request):
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        ip = x_forwarded_for.split(",")[0] if x_forwarded_for else request.META.get("REMOTE_ADDR")
        return ip
