"""URL Configuration for Hydraulic Diagnostics Backend.

Django Unfold handles admin customization via settings.py UNFOLD config.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/support/", include("apps.support.urls")),
    path("admin/docs/", include("apps.docs.urls")),
    path("health/", include("apps.monitoring.urls")),
]

# Serve media and static files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
