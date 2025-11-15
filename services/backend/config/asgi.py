"""ASGI config for Hydraulic Diagnostics Backend.

Exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see:
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/

Python 3.14+ with full async/await support.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from django.core.asgi import get_asgi_application

if TYPE_CHECKING:
    from django.core.handlers.asgi import ASGIHandler

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Get ASGI application
application: ASGIHandler = get_asgi_application()

# Optional: Wrap with additional ASGI middleware
# Example: Sentry ASGI integration, CORS, etc.
# from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
# application = SentryAsgiMiddleware(application)