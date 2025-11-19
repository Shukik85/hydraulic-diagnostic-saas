"""Monitoring views for health checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import connection
from django.http import JsonResponse

if TYPE_CHECKING:
    from django.http import HttpRequest


def health_check(request: HttpRequest) -> JsonResponse:  # noqa: ARG001
    """Health check endpoint for Docker.

    Args:
        request: HTTP request (unused, required by Django URL routing)

    Returns:
        JsonResponse with status and database connection state
    """
    try:
        # Check database connection
        connection.ensure_connection()
        return JsonResponse({"status": "ok", "database": "connected"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
