# apps/rag_assistant/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, RagSystemViewSet, RagQueryLogViewSet, TaskStatusView

router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='documents')
router.register(r'systems', RagSystemViewSet, basename='systems')
router.register(r'logs', RagQueryLogViewSet, basename='logs')

urlpatterns = [
    path('', include(router.urls)),
    path('tasks/<str:task_id>/', TaskStatusView.as_view(), name='task-status'),
]
