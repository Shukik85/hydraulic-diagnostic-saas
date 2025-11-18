"""
Environment callback — Production/Staging/Development badge в админке.
"""

import os

from django.utils.translation import gettext_lazy as _


def environment_callback(request):  # noqa: ARG001
    """
    Возвращает environment badge для Unfold.
    Отображается в top-right corner админки.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    environments = {
        "production": {
            "label": _("🔴 PRODUCTION"),
            "color": "danger",  # Red
        },
        "staging": {
            "label": _("🟡 STAGING"),
            "color": "warning",  # Amber
        },
        "development": {
            "label": _("🟢 DEV"),
            "color": "success",  # Green
        },
    }

    return environments.get(
        env,
        {
            "label": _(f"🔵 {env.upper()}"),
            "color": "info",
        },
    )
