from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    HydraulicSystemViewSet, 
    SensorDataViewSet, 
    DiagnosticReportViewSet,
)

# Создание роутера для ViewSets
router = DefaultRouter()
router.register(r'hydraulic-systems', HydraulicSystemViewSet, basename='hydraulic-system')
router.register(r'sensor-data', SensorDataViewSet, basename='sensor-data')
router.register(r'diagnostic-reports', DiagnosticReportViewSet, basename='diagnostic-report')

# URL patterns
urlpatterns = [
    # API endpoints через router
    path('', include(router.urls)),
    
    # Дополнительные endpoints для специфических операций
    # (могут быть добавлены в будущем)
]

# Именованные URL patterns для удобства
app_name = 'diagnostics'
