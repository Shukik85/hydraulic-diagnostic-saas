"""Core utility functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django.http import HttpRequest


def environment_callback(request: HttpRequest) -> str:
    """Return current environment name for Unfold admin header.
    
    Shows environment badge (DEVELOPMENT, STAGING, PRODUCTION) in admin.
    """
    env = os.getenv("ENVIRONMENT", "development")
    return env.upper()
