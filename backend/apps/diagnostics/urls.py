# apps/diagnostics/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    HydraulicSystemViewSet,
    SystemComponentViewSet,
    SensorDataViewSet,
    DiagnosticReportViewSet,
    MaintenanceScheduleViewSet,
)

router = DefaultRouter()
router.register(r'systems', HydraulicSystemViewSet, basename='systems')
router.register(r'components', SystemComponentViewSet, basename='components')
router.register(r'sensor-data', SensorDataViewSet, basename='sensor-data')
router.register(r'reports', DiagnosticReportViewSet, basename='reports')
router.register(r'maintenance', MaintenanceScheduleViewSet, basename='maintenance')

urlpatterns = [
    path('', include(router.urls)),
]
