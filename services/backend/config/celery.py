"""Celery configuration for Hydraulic Diagnostics Backend.

Async task queue configuration with beat scheduler.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("hydraulic_diagnostics")

# Load config from Django settings with 'CELERY_' prefix
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks in all installed apps
app.autodiscover_tasks()

# Celery Beat schedule for periodic tasks
app.conf.beat_schedule = {
    # Support Module: SLA breach monitoring
    "check-sla-breaches": {
        "task": "apps.support.tasks.check_sla_breaches",
        "schedule": crontab(minute="*/30"),  # Every 30 minutes
    },
    # Support Module: Auto-assign new tickets
    "auto-assign-tickets": {
        "task": "apps.support.tasks.auto_assign_tickets",
        "schedule": crontab(minute="*/15"),  # Every 15 minutes
    },
    # Users Module: Check trial expirations (if exists)
    # "check-trial-expirations": {
    #     "task": "apps.users.tasks.check_trial_expirations",
    #     "schedule": crontab(hour="0", minute="0"),  # Daily at midnight
    # },
}


@app.task(bind=True, ignore_result=True)
def debug_task(self) -> None:
    """Debug task for testing Celery configuration."""
    print(f"Request: {self.request!r}")
