"""
Equipment API URLs
"""
from django.urls import path
from .views import EquipmentConfigView, SensorInferenceView

urlpatterns = [
    path("equipment/config/", EquipmentConfigView.as_view()),
    path("equipment/config/<str:equipment_id>/", EquipmentConfigView.as_view()),
    path("inference/", SensorInferenceView.as_view()),
    path("inference/result/<str:task_id>/", SensorInferenceView.as_view()),
]
