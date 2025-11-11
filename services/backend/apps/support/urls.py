"""
Support URLs (quick actions API)
"""
from django.urls import path
from . import views

urlpatterns = [
    path('reset-password/<uuid:user_id>/', views.reset_password, name='reset_password'),
    path('extend-trial/<uuid:user_id>/', views.extend_trial, name='extend_trial'),
]
