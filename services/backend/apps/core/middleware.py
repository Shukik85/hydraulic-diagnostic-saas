from collections.abc import Callable

from django.core.cache import cache
from django.http import JsonResponse


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
