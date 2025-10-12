from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls), # Админка Django
    path('api/', include('apps.diagnostics.urls')), # Добавляем маршруты приложения diagnostics
    path('api/', include('apps.users.urls')), # Добавляем маршруты приложения users
]
