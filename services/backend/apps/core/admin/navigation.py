"""
Django Unfold Sidebar Navigation — Enterprise struktura.
Sekcie: Operations, ML & Diagnostics, Business, Users & Auth.
"""

from django.urls import reverse_lazy
from django.utils.translation import gettext_lazy as _


def get_sidebar_navigation() -> list:
    """Возвращает конфигурацию sidebar для Unfold admin."""
    return [
        # =====================================================================
        # OPERATIONS
        # =====================================================================
        {
            "title": _("🎛️ Operations"),
            "separator": True,
            "collapsible": False,
            "items": [
                {
                    "title": _("Dashboard"),
                    "icon": "dashboard",
                    "link": reverse_lazy("admin:index"),
                },
                {
                    "title": _("Equipment"),
                    "icon": "precision_manufacturing",
                    "link": reverse_lazy("admin:equipment_equipment_changelist"),
                },
                {
                    "title": _("Monitoring"),
                    "icon": "monitor_heart",
                    "link": reverse_lazy("admin:monitoring_apilog_changelist"),
                },
            ],
        },
        # =====================================================================
        # ML & DIAGNOSTICS
        # =====================================================================
        {
            "title": _("🧠 ML & Diagnostics"),
            "separator": True,
            "collapsible": False,
            "items": [
                {
                    "title": _("GNN Models"),
                    "icon": "hub",
                    "link": reverse_lazy("admin:gnn_config_gnnmodelconfig_changelist"),
                },
                {
                    "title": _("Predictions"),
                    "icon": "troubleshoot",
                    "link": reverse_lazy("admin:gnn_config_predictionresult_changelist"),
                },
            ],
        },
        # =====================================================================
        # BUSINESS
        # =====================================================================
        {
            "title": _("💼 Business"),
            "separator": True,
            "collapsible": True,
            "items": [
                {
                    "title": _("Subscriptions"),
                    "icon": "credit_card",
                    "link": reverse_lazy("admin:subscriptions_subscription_changelist"),
                },
                {
                    "title": _("Support Tickets"),
                    "icon": "support_agent",
                    "link": reverse_lazy("admin:support_supportticket_changelist"),
                },
            ],
        },
        # =====================================================================
        # USERS & AUTH
        # =====================================================================
        {
            "title": _("👥 Users & Auth"),
            "separator": True,
            "collapsible": True,
            "items": [
                {
                    "title": _("Users"),
                    "icon": "person",
                    "link": reverse_lazy("admin:users_user_changelist"),
                },
                {
                    "title": _("Notifications"),
                    "icon": "notifications",
                    "link": reverse_lazy("admin:notifications_notification_changelist"),
                },
                {
                    "title": _("Docs"),
                    "icon": "description",
                    "link": reverse_lazy("admin:docs_documentation_changelist"),
                },
            ],
        },
    ]
