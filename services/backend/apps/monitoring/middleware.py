"""
Request logging middleware
"""

import time

from .models import APILog


class RequestLoggingMiddleware:
    """Log all API requests"""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()

        response = self.get_response(request)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Extract user ID if authenticated
        user_id = request.user.id if request.user.is_authenticated else None

        # Log to database (async recommended for production)
        APILog.objects.create(
            user_id=user_id,
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            ip_address=self.get_client_ip(request),
            user_agent=request.META.get("HTTP_USER_AGENT", "")[:512],
        )

        return response

    def get_client_ip(self, request):
        """Extract client IP address"""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        return (
            x_forwarded_for.split(",")[0]
            if x_forwarded_for
            else request.META.get("REMOTE_ADDR")
        )
