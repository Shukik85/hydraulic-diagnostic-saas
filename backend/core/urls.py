from django.contrib import admin
from django.urls import path, include
<<<<<<< HEAD
from django.conf import settings
from django.conf.urls.static import static
from apps.users.views import UserRegistrationView, UserLoginView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/register/', UserRegistrationView.as_view(), name='register'),
    path('api/auth/login/', UserLoginView.as_view(),
         name='token_obtain_pair'),
    path('api/auth/token/refresh/',
         TokenRefreshView.as_view(), name='token_refresh'),
    path('api/users/', include('apps.users.urls')),
    path('api/diagnostics/', include('apps.diagnostics.urls')),
    path('api/rag/', include('apps.rag_assistant.urls')),
=======

urlpatterns = [
    path('admin/', admin.site.urls), # Админка Django
    path('api/', include('apps.diagnostics.urls')), # Добавляем маршруты приложения diagnostics
    path('api/', include('apps.users.urls')), # Добавляем маршруты приложения users
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
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
