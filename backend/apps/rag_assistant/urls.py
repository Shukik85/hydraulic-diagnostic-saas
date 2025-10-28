"""Production RAG Assistant URLs with DRF routing and monitoring."""

from django.urls import include, path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.routers import DefaultRouter

from .views import (
    DocumentViewSet,
    HealthCheckView,
    RagQueryLogViewSet,
    RagSystemViewSet,
)

# Создаём router и регистрируем production ViewSets
router = DefaultRouter()
router.register(r"systems", RagSystemViewSet, basename="ragsystem")
router.register(r"documents", DocumentViewSet, basename="document")
router.register(r"logs", RagQueryLogViewSet, basename="ragquerylog")


# API метрики для мониторинга
@api_view(["GET"])
def api_metrics(request):
    """Получение метрик API для мониторинга."""
    import time

    from django.core.cache import cache
    from .models import Document, RagQueryLog, RagSystem

    current_hour = time.strftime("%Y-%m-%d:%H")
    current_day = time.strftime("%Y-%m-%d")

    # Получаем метрики из кэша и БД
    metrics = {
        "systems": {
            "total": RagSystem.objects.count(),
            "active_today": RagQueryLog.objects.filter(
                timestamp__date=current_day.split(":")[0]
            ).values("system").distinct().count(),
        },
        "documents": {
            "total": Document.objects.count(),
            "by_language": dict(
                Document.objects.values("language").annotate(
                    count=models.Count("id")
                ).values_list("language", "count")
            ),
        },
        "requests": {
            "current_hour": cache.get(
                f"performance_metrics:{current_hour}:requests", 0
            ),
            "today": cache.get(f"performance_metrics:{current_day}:requests", 0),
        },
        "performance": {
            "avg_response_time_ms": cache.get(
                f"performance_metrics:{current_hour}:avg_duration", 0
            ),
        },
        "timestamp": time.time(),
    }

    return Response(metrics)


urlpatterns = [
    path("health/", HealthCheckView.as_view(), name="rag-health"),
    path("metrics/", api_metrics, name="rag-metrics"),
    path("", include(router.urls)),
]
