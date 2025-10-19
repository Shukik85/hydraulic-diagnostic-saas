from django.urls import include, path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.routers import DefaultRouter

# Импорт ОПТИМИЗИРОВАННЫХ ViewSets
from .optimized_views import (
    OptimizedDocumentViewSet,
    OptimizedRagQueryLogViewSet,
    OptimizedRagSystemViewSet,
)

# Импорт обычных views как fallback


# Создаем router и регистрируем ОПТИМИЗИРОВАННЫЕ ViewSets
router = DefaultRouter()
router.register(r"documents", OptimizedDocumentViewSet, basename="document")
router.register(r"systems", OptimizedRagSystemViewSet, basename="ragsystem")
router.register(r"query-logs", OptimizedRagQueryLogViewSet, basename="ragquerylog")


# API метрики для мониторинга
@api_view(["GET"])
def api_metrics(request):
    """
    Получение метрик API для мониторинга
    """
    import time

    from django.core.cache import cache

    current_hour = time.strftime("%Y-%m-%d:%H")
    current_day = time.strftime("%Y-%m-%d")

    metrics = {
        "requests": {
            "current_hour": cache.get(
                f"performance_metrics:{current_hour}:requests", 0
            ),
            "today": cache.get(f"performance_metrics:{current_day}:requests", 0),
        },
        "ai_operations": {
            "current_hour": cache.get(f"ai_metrics:{current_hour}:ai_requests", 0),
            "slow_operations_hour": cache.get(
                f"ai_metrics:{current_hour}:slow_ai_requests", 0
            ),
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
    path("api/metrics/", api_metrics, name="api-metrics"),  # Метрики API
    path("", include(router.urls)),
]
