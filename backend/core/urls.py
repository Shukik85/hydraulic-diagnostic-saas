from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),  # Админка Django
    # JWT Authentication endpoints
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # Application routes
    path('api/users/', include('apps.users.urls')),  # Добавляем маршруты приложения users
    path('api/diagnostics/', include('apps.diagnostics.urls')),  # Добавляем маршруты приложения diagnostics
    path('api/rag/', include('apps.rag_assistant.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)

# Настройка админ панели
admin.site.site_header = 'Гидравлическая диагностика'
admin.site.site_title = 'Админ панель'
admin.site.index_title = 'Управление системой'
