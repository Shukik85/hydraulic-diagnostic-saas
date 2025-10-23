"""Модуль проекта с автогенерированным докстрингом."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

# Импорт health checks
from .health_checks import health_check, liveness_check, readiness_check

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    # Health Checks - КРИТИЧНО для production
    path("health/", health_check, name="health-check"),
    path("readiness/", readiness_check, name="readiness"),
    path("liveness/", liveness_check, name="liveness"),
    # JWT Authentication
    path("api/auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),
    # API endpoints
    path("api/users/", include("apps.users.urls")),
    path("api/diagnostics/", include("apps.diagnostics.urls")),
    path("api/rag_assistant/", include("apps.rag_assistant.urls")),
]

# Static files serving in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

    # Django Debug Toolbar
    if "debug_toolbar" in settings.INSTALLED_APPS:
        import debug_toolbar

        urlpatterns = [
            path("__debug__/", include(debug_toolbar.urls)),
        ] + urlpatterns

# Custom error handlers for production
if not settings.DEBUG:
    from django.http import JsonResponse

    def handler404(request, exception):
        return JsonResponse(
            {
                "error": "Not Found",
                "status_code": 404,
                "message": "The requested resource was not found.",
            },
            status=404,
        )

    def handler500(request):
        return JsonResponse(
            {
                "error": "Internal Server Error",
                "status_code": 500,
                "message": "An internal server error occurred.",
            },
            status=500,
        )
