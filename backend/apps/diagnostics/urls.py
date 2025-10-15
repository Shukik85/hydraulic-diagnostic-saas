from django.urls import path, include
from rest_framework.routers import DefaultRouter
<<<<<<< HEAD
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
=======
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .views import HydraulicSystemViewSet, SensorDataViewSet, DiagnosticReportViewSet

@api_view(['GET'])
def api_root(request):
    """Root API endpoint"""
    return Response({
        'message': 'Hydraulic Diagnostic System API',
        'version': '1.0',
        'endpoints': {
            'hydraulic_systems': '/api/hydraulic-systems/',
            'sensor_data': '/api/sensor-data/',
            'diagnostic_reports': '/api/diagnostic-reports/',
            'auth_login': '/api/auth/login/',
            'auth_register': '/api/auth/register/',
            'auth_profile': '/api/auth/profile/',
        }
    })

router = DefaultRouter()
router.register(r'hydraulic-systems', HydraulicSystemViewSet, basename='hydraulicsystem')
router.register(r'sensor-data', SensorDataViewSet, basename='sensordata')
router.register(r'diagnostic-reports', DiagnosticReportViewSet, basename='diagnosticreport')

urlpatterns = [
    path('', api_root, name='api-root'),  # Добавляем root endpoint
    path('', include(router.urls)),
]
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
