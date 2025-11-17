"""Core admin utilities for Django Unfold."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models import Count, Q
from django.utils import timezone
from unfold.widgets import UnfoldAdminTextInputWidget

if TYPE_CHECKING:
    from django.http import HttpRequest


def dashboard_callback(request: HttpRequest, context: dict) -> list[dict]:
    """Custom dashboard widgets for Unfold admin.
    
    Returns list of widget configurations for the dashboard.
    """
    from apps.equipment.models import Equipment
    from apps.gnn_config.models import GNNModelConfiguration
    from apps.support.models import SupportTicket
    from apps.users.models import User

    # Calculate metrics
    total_users = User.objects.count()
    active_equipment = Equipment.objects.filter(is_active=True).count()
    open_tickets = SupportTicket.objects.filter(
        status__in=["new", "open", "in_progress"]
    ).count()
    active_models = GNNModelConfiguration.objects.filter(is_active=True).count()

    return [
        {
            "type": "metric",
            "title": "Пользователи",
            "metric": total_users,
            "footer": "Всего в системе",
        },
        {
            "type": "metric",
            "title": "Оборудование",
            "metric": active_equipment,
            "footer": "Активных систем",
        },
        {
            "type": "metric",
            "title": "Поддержка",
            "metric": open_tickets,
            "footer": "Открытых тикетов",
            "color": "orange" if open_tickets > 5 else "green",
        },
        {
            "type": "metric",
            "title": "GNN Модели",
            "metric": active_models,
            "footer": "Активных моделей",
        },
    ]


def environment_callback(request: HttpRequest) -> str:
    """Return current environment name for Unfold header.
    
    Shows environment badge in admin header.
    """
    import os

    env = os.getenv("ENVIRONMENT", "development")
    return env.upper()
