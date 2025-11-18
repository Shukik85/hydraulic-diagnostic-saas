"""
Django Unfold Dashboard Callback — KPI виджеты + метрики.
Teal/Steel metallic theme с материал-иконками.
"""

from django.utils.translation import gettext_lazy as _


def dashboard_callback(request, context):
    """
    Custom dashboard для Unfold admin.

    Возвращает KPI карточки и готовит контекст для виджетов.
    """
    from apps.equipment.models import Equipment
    from apps.gnn_config.models import GNNModelConfig
    from apps.support.models import SupportTicket
    from apps.users.models import User

    try:
        total_equipment = Equipment.objects.count()
    except Exception:
        total_equipment = 0

    try:
        active_models = GNNModelConfig.objects.filter(is_active=True).count()
    except Exception:
        active_models = 0

    try:
        open_tickets = SupportTicket.objects.filter(
            status__in=["new", "open", "in_progress"]
        ).count()
    except Exception:
        open_tickets = 0

    try:
        total_users = User.objects.filter(is_active=True).count()
    except Exception:
        total_users = 0

    # KPI виджеты (Unfold-compatible)
    context.update(
        {
            "kpi": [
                {
                    "title": _("Active Systems"),
                    "metric": total_equipment,
                    "footer": _("Hydraulic equipment monitored"),
                    "icon": "precision_manufacturing",
                },
                {
                    "title": _("ML Models"),
                    "metric": active_models,
                    "footer": _("Active GNN models"),
                    "icon": "hub",
                },
                {
                    "title": _("Open Tickets"),
                    "metric": open_tickets,
                    "footer": _("Requires attention"),
                    "icon": "support_agent",
                    "color": "warning" if open_tickets > 5 else "success",
                },
                {
                    "title": _("Platform Users"),
                    "metric": total_users,
                    "footer": _("Active users"),
                    "icon": "person",
                },
            ]
        }
    )

    return context
