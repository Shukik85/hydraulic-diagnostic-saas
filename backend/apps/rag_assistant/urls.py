from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'knowledge-base', views.KnowledgeBaseViewSet, basename='knowledgebase')
router.register(r'queries', views.RAGQueryViewSet, basename='ragquery')
router.register(r'conversations', views.RAGConversationViewSet, basename='ragconversation')
router.register(r'settings', views.RAGSystemSettingsViewSet, basename='ragsettings')

urlpatterns = [
    path('', include(router.urls)),
    path('stats/', views.rag_system_stats, name='rag-stats'),
]
