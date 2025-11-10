"""
URL routing for ML prediction API
"""
from django.urls import path
from . import api_predict

app_name = 'predict'

urlpatterns = [
    path('predict/', api_predict.predict_anomaly, name='predict_anomaly'),
    path('ml/health/', api_predict.ml_service_health, name='ml_health'),
]
