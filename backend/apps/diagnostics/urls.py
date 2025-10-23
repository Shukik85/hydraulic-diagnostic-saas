"""Модуль проекта с автогенерированным докстрингом."""

# apps/diagnostics/urls.py

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    DiagnosticReportViewSet,
    HydraulicSystemViewSet,
    SensorDataViewSet,
    SystemComponentViewSet,
)

router = DefaultRouter()
router.register(r"systems", HydraulicSystemViewSet, basename="systems")
router.register(r"components", SystemComponentViewSet, basename="components")
router.register(r"sensor-data", SensorDataViewSet, basename="sensor-data")
router.register(r"reports", DiagnosticReportViewSet, basename="reports")

urlpatterns = [
    path("", include(router.urls)),
]
