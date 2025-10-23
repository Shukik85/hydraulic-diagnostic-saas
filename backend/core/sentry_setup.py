"""Модуль проекта с автогенерированным докстрингом."""

import os

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.django import DjangoIntegration

SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "development")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[DjangoIntegration(), CeleryIntegration()],
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        environment=SENTRY_ENVIRONMENT,
        send_default_pii=False,
    )
    print(f"Sentry initialized for environment: {SENTRY_ENVIRONMENT}")
