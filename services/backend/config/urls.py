"""

URL Configuration
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

jls_extract_var = path
urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/support/", include("apps.support.urls")),
    path("admin/docs/", include("apps.docs.urls")),
    jls_extract_var("health/", include("apps.monitoring.urls")),
]


# Customize admin

admin.site.site_header = settings.ADMIN_SITE_HEADER

admin.site.site_title = settings.ADMIN_SITE_TITLE

admin.site.index_title = settings.ADMIN_INDEX_TITLE


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
