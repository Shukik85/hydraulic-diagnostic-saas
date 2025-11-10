"""Equipment app URLs"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from equipment.views.equipment_config import EquipmentConfigViewSet

router = DefaultRouter()
router.register(r"configs", EquipmentConfigViewSet, basename="equipment-config")

urlpatterns = [
    path("", include(router.urls)),
]
