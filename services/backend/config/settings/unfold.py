"""
Django Unfold Admin Theme Configuration.

Metallic Industrial Theme для Hydraulic Diagnostic Platform.
Все UI/UX настройки, цвета, навигация, виджеты, callbacks.

Используется в config/settings.py:
    from config.settings.unfold import UNFOLD
"""  # noqa: RUF002

from django.utils.translation import gettext_lazy as _

# Импорт callbacks
# Импорт навигации
from apps.core.admin.navigation import get_sidebar_navigation

# Импорт палитры
from apps.core.theme.colors import METALLIC_COLORS

# =============================================================================
# UNFOLD CONFIGURATION — PRODUCTION-READY
# =============================================================================

UNFOLD = {
    # =========================================================================
    # BRANDING & IDENTITY
    # =========================================================================
    "SITE_TITLE": _("Hydraulic Diagnostics"),
    "SITE_HEADER": _("Hydraulic Diagnostic Platform"),
    "SITE_URL": "/",
    "SITE_SYMBOL": "precision_manufacturing",  # Material Icons
    # Логотип (если будут файлы)
    # "SITE_ICON": {
    #     "light": "/static/admin/img/logo-light.svg",
    #     "dark": "/static/admin/img/logo-dark.svg",
    # },
    # =========================================================================
    # COLORS — Metallic Teal/Steel Theme
    # =========================================================================
    "COLORS": METALLIC_COLORS,
    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    "SIDEBAR": {
        "show_search": True,
        "show_all_applications": False,
        "navigation": get_sidebar_navigation(),
    },
    # =========================================================================
    # DASHBOARD & ENVIRONMENT
    # =========================================================================
    "DASHBOARD_CALLBACK": "apps.core.admin.dashboard.dashboard_callback",
    "ENVIRONMENT": "apps.core.admin.environment.environment_callback",
    # =========================================================================
    # EXTENSIONS
    # =========================================================================
    "EXTENSIONS": {
        "modeltranslation": {
            "flags": {
                "en": "🇬🇧",
                "ru": "🇷🇺",
            },
        },
    },
    # =========================================================================
    # UI THEME
    # =========================================================================
    "THEME": "dark",  # "light", "dark", "auto"
}
