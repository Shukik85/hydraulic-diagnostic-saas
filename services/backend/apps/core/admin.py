"""Core admin utilities for Django Unfold."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django.http import HttpRequest

def dashboard_callback(request: HttpRequest, context: dict) -> dict:
    """Custom dashboard widgets for Unfold admin.
    Safe to call even if tables don't exist yet (during initial setup).
    Returns dict with 'widgets' key for Unfold compatibility.
    """
    try:
        from apps.equipment.models import Equipment
        from apps.gnn_config.models import GNNModelConfig
        from apps.support.models import SupportTicket
        from apps.users.models import User

        # Calculate metrics (with error handling for missing tables)
        total_users = User.objects.count()
        try:
            active_equipment = Equipment.objects.count()
        except Exception:
            active_equipment = 0
        try:
            open_tickets = SupportTicket.objects.filter(status__in=["new", "open", "in_progress"]).count()
        except Exception:
            open_tickets = 0
        try:
            active_models = GNNModelConfig.objects.filter(is_active=True).count()
        except Exception:
            active_models = 0
        return {
            "widgets": [
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
                    "footer": "Зарегистрировано систем",
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
        }
    except Exception:
        # If something goes wrong, return empty widgets to prevent admin crash
        return {"widgets": []}
