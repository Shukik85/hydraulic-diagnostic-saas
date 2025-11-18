"""Django settings for Hydraulic Diagnostics Backend.

Production-ready configuration with:
- Python 3.14+ support
- ASGI/async middleware
- Environment-based configuration
- Security hardening
- Django Unfold admin theme

For deployment guide, see: services/backend/README.md
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Загрузка .env
load_dotenv(BASE_DIR / ".env")

# ============================================================
# SECURITY
# ============================================================
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "django-secret-key-change-in-production")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
ALLOWED_HOSTS: list[str] = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# ============================================================
# APPLICATION DEFINITION
# ============================================================
INSTALLED_APPS = [
    # Django Unfold - modern admin theme (MUST be before django.contrib.admin)
    "unfold",
    "unfold.contrib.filters",
    "unfold.contrib.forms",
    # Django core
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "rest_framework_simplejwt",
    "corsheaders",
    "drf_spectacular",
    "django_celery_beat",
    "django_celery_results",
    "django_prometheus",
    # Local apps
    "apps.core",
    "apps.users",
    "apps.subscriptions",
    "apps.gnn_config",
    "apps.equipment",
    "apps.notifications",
    "apps.monitoring",
    "apps.support",
    "apps.docs",
]

MIDDLEWARE = [
    "django_prometheus.middleware.PrometheusBeforeMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "apps.core.middleware.RateLimitMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

# ============================================================
# ASGI/WSGI CONFIGURATION
# ============================================================
ASGI_APPLICATION = "config.asgi.application"
WSGI_APPLICATION = "config.wsgi.application"

# ============================================================
# TEMPLATES
# ============================================================
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.static",
                "django.template.context_processors.media",
                "django.template.context_processors.i18n",
                "django.template.context_processors.tz",
            ],
        },
    },
]

# ============================================================
# DATABASE
# ============================================================
if DEBUG:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
    print("🔧 Development mode: Using SQLite database")
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("DATABASE_NAME", "hydraulic_db"),
            "USER": os.getenv("DATABASE_USER", "postgres"),
            "PASSWORD": os.getenv("DATABASE_PASSWORD", "postgres"),
            "HOST": os.getenv("DATABASE_HOST", "postgres"),
            "PORT": os.getenv("DATABASE_PORT", "5432"),
            "CONN_MAX_AGE": 600,
            "OPTIONS": {
                "connect_timeout": 10,
            },
        }
    }
    print("🚀 Production mode: Using PostgreSQL database")

# ============================================================
# AUTHENTICATION
# ============================================================
AUTH_USER_MODEL = "users.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 8},
    },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
]

# ============================================================
# INTERNATIONALIZATION
# ============================================================
LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_L10N = True
USE_TZ = True

LANGUAGES = [
    ("ru", "Русский"),
    ("en", "English"),
]

LOCALE_PATHS = [BASE_DIR / "locale"]

# ============================================================
# STATIC FILES
# ============================================================
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# ============================================================
# REST FRAMEWORK
# ============================================================
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 50,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
}

# JWT Settings
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=1),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
}

# API Documentation
SPECTACULAR_SETTINGS = {
    "TITLE": "Hydraulic Diagnostics API",
    "DESCRIPTION": "Backend API for Hydraulic Diagnostics SaaS",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# ============================================================
# CELERY CONFIGURATION
# ============================================================
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/1")
CELERY_RESULT_BACKEND = "django-db"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
CELERY_TASK_SOFT_TIME_LIMIT = 300
CELERY_TASK_TIME_LIMIT = 360

# ============================================================
# EMAIL CONFIGURATION
# ============================================================
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.sendgrid.net")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "apikey")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "")
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "noreply@hydraulic-diagnostics.com")
SERVER_EMAIL = DEFAULT_FROM_EMAIL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# ============================================================
# STRIPE CONFIGURATION
# ============================================================
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# ============================================================
# CORS CONFIGURATION
# ============================================================
CORS_ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")
CORS_ALLOW_CREDENTIALS = True

# ============================================================
# SECURITY SETTINGS (Production)
# ============================================================
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    X_FRAME_OPTIONS = "DENY"

# ============================================================
# LOGGING
# ============================================================
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("LOG_LEVEL", "INFO"),
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}

# ============================================================
# DEFAULT PRIMARY KEY
# ============================================================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ============================================================
# DJANGO UNFOLD CONFIGURATION
# ============================================================
# from django.urls import reverse_lazy
# from django.utils.translation import gettext_lazy as _

# UNFOLD = {
#     "SITE_TITLE": "Hydraulic Diagnostics",
#     "SITE_HEADER": "Hydraulic Diagnostics",
#     "SITE_URL": "/",
#     # Навигация в сайдбаре
#     "SIDEBAR": {
#         "show_search": True,
#         "show_all_applications": False,
#         "navigation": [
#             {
#                 "title": _("Основное"),
#                 "separator": True,
#                 "items": [
#                     {
#                         "title": _("Dashboard"),
#                         "icon": "dashboard",
#                         "link": reverse_lazy("admin:index"),
#                     },
#                     {
#                         "title": _("Пользователи"),
#                         "icon": "people",
#                         "link": reverse_lazy("admin:users_user_changelist"),
#                     },
#                     {
#                         "title": _("Поддержка"),
#                         "icon": "support_agent",
#                         "link": reverse_lazy("admin:support_supportticket_changelist"),
#                     },
#                 ],
#             },
#             {
#                 "title": _("Система"),
#                 "separator": True,
#                 "collapsible": True,
#                 "items": [
#                     {
#                         "title": _("Оборудование"),
#                         "icon": "settings_input_component",
#                         "link": reverse_lazy("admin:equipment_equipment_changelist"),
#                     },
#                     {
#                         "title": _("GNN Модели"),
#                         "icon": "psychology",
#                         "link": reverse_lazy("admin:gnn_config_gnnmodelconfig_changelist"),
#                     },
#                     {
#                         "title": _("Логи API"),
#                         "icon": "api",
#                         "link": reverse_lazy("admin:monitoring_apilog_changelist"),
#                     },
#                     {
#                         "title": _("Уведомления"),
#                         "icon": "notifications",
#                         "link": reverse_lazy("admin:notifications_notification_changelist"),
#                     },
#                     {
#                         "title": _("Подписки"),
#                         "icon": "credit_card",
#                         "link": reverse_lazy("admin:subscriptions_subscription_changelist"),
#                     },
#                 ],
#             },
#         ],
#     },
#     # Дополнительные настройки
#     "ENVIRONMENT": "apps.core.utils.environment_callback",
#     # "DASHBOARD_CALLBACK": "apps.core.admin.dashboard_callback",
# }

# ============================================================
# SENTRY
# ============================================================
SENTRY_DSN = os.getenv("SENTRY_DSN", "")

if not DEBUG and SENTRY_DSN and SENTRY_DSN.startswith("https://"):
    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.django import DjangoIntegration
    from sentry_sdk.integrations.redis import RedisIntegration

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            DjangoIntegration(),
            CeleryIntegration(),
            RedisIntegration(),
        ],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
        environment=os.getenv("ENVIRONMENT", "production"),
    )
    print("✓ Sentry error tracking enabled")
else:
    print("🆎  Sentry disabled in development mode" if DEBUG else "⚠️  Sentry DSN not configured")
from config.settings.unfold import UNFOLD  # noqa: E402

print(f"UNFOLD config loaded: {len(UNFOLD)} keys")
# Явно экспортируем для других модулей
__all__ = [
    ...,
    "UNFOLD",
]
