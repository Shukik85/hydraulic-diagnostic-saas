"""Модуль проекта с автогенерированным докстрингом."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Если Channels и diagnostics.routing доступны — используем их, иначе чистый ASGI
try:
    from channels.auth import AuthMiddlewareStack  # type: ignore
    from channels.routing import ProtocolTypeRouter, URLRouter  # type: ignore

    from apps.diagnostics import routing  # type: ignore

    application = ProtocolTypeRouter(
        {
            "http": get_asgi_application(),
            "websocket": AuthMiddlewareStack(URLRouter(routing.websocket_urlpatterns)),
        }
    )
except Exception:  # pragma: no cover
    # Fallback: только HTTP ASGI приложение
    application = get_asgi_application()
