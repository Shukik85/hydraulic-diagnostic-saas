"""Custom Django Admin Site Configuration.

Makes admin interface more user-friendly with:
- Custom branding
- Russian language
- Improved navigation
- Custom dashboard
"""

from django.contrib import admin
from django.contrib.admin import AdminSite
from django.utils.translation import gettext_lazy as _


class HydraulicAdminSite(AdminSite):
    """Custom admin site with friendly branding."""
    
    # Header & Title
    site_header = _("🔧 Hydraulic Diagnostics - Панель управления")
    site_title = _("Hydraulic Admin")
    index_title = _("Добро пожаловать в систему управления")
    
    # Enable navigation sidebar
    enable_nav_sidebar = True
    
    def each_context(self, request):
        """Add custom context to all admin pages."""
        context = super().each_context(request)
        context.update({
            'site_url': '/',
            'has_permission': request.user.is_active,
        })
        return context


# Replace default admin site
admin.site = HydraulicAdminSite()
admin.sites.site = admin.site
