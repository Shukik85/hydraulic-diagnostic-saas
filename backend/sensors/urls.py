"""
URL Configuration for Sensors API.

Defines REST API endpoints for sensor data collection and monitoring.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    SensorNodeViewSet,
    SensorConfigViewSet,
    SensorReadingViewSet,
    sensor_health_status
)

# Create DRF router
router = DefaultRouter()
router.register(r'nodes', SensorNodeViewSet)
router.register(r'configs', SensorConfigViewSet)
router.register(r'readings', SensorReadingViewSet)

app_name = 'sensors'

urlpatterns = [
    # DRF ViewSet endpoints
    path('api/sensors/', include(router.urls)),
    
    # Health check endpoint (for monitoring)
    path('health/sensors/', sensor_health_status, name='sensor_health'),
    
    # Specific endpoint patterns (if needed)
    # path('api/sensors/nodes/<int:pk>/poll/', SensorNodeViewSet.as_view({'post': 'poll_now'})),
    # path('api/sensors/readings/latest/', SensorReadingViewSet.as_view({'get': 'latest'})),
]
