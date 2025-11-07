"""Django settings for hydraulic-diagnostic-saas project.

CI-compatible with TimescaleDB and Celery Beat integration.
Django 5.1 + DRF 3.15 compatible.
"""

import os
from datetime import timedelta
from pathlib import Path

import structlog
from celery.schedules import crontab
from corsheaders.defaults import default_headers
from decouple import Csv, config

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Core & Environment
# ----------------------------------------------------------------------------
SECRET_KEY = config("SECRET_KEY")
DEBUG = config("DEBUG", default=False, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="", cast=Csv()) or [
    "localhost",
    "127.0.0.1",
]
CSRF_TRUSTED_ORIGINS = config("CSRF_TRUSTED_ORIGINS", default="", cast=Csv()) or []

# ----------------------------------------------------------------------------
# Applications
# ----------------------------------------------------------------------------
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",  # ADDED: JWT blacklist
    "drf_spectacular",  # OpenAPI/Swagger documentation
    "corsheaders",
    "django_filters",
    "django_extensions",
    "django_celery_beat",  # Database-backed periodic tasks for TimescaleDB
]

# FIXED: Исправлено на реальную структуру проекта
LOCAL_APPS = [
    "users.apps.UsersConfig",           # backend/users/
    "sensors.apps.SensorsConfig",       # backend/sensors/
    "diagnostics.apps.DiagnosticsConfig",  # backend/diagnostics/ (FIXED)
    "rag_assistant.apps.RagAssistantConfig",  # backend/rag_assistant/
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# ----------------------------------------------------------------------------
# Middleware (order matters)
# ----------------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ----------------------------------------------------------------------------
# ASGI / WSGI
# ----------------------------------------------------------------------------
WSGI_APPLICATION = "core.wsgi.application"
ASGI_APPLICATION = "core.asgi.application"

# ----------------------------------------------------------------------------
# Database: PostgreSQL with TimescaleDB support
# ----------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": config("DATABASE_NAME"),
        "USER": config("DATABASE_USER"),
        "PASSWORD": config("DATABASE_PASSWORD"),
        "HOST": config("DATABASE_HOST", default="db"),
        "PORT": config("DATABASE_PORT", default=5432, cast=int),
        "CONN_MAX_AGE": 0,  # required for pooling
        "OPTIONS": {
            "pool": {
                "min_size": 2,
                "max_size": 20,
                "max_idle": 60,
                "timeout": 30,
                "max_lifetime": 300,
            }
        },
    }
}

# ----------------------------------------------------------------------------
# Cache: Redis
# ----------------------------------------------------------------------------
REDIS_URL = config("REDIS_URL", default="redis://redis:6379/1")
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "client_class": "django_redis.client.DefaultClient",
            "retry_on_timeout": True,
        },
        "TIMEOUT": 300,
        "KEY_PREFIX": "hdx",
    }
}

SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"

# ----------------------------------------------------------------------------
# Static & Media
# ----------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ----------------------------------------------------------------------------
# Security Headers & HTTPS
# ----------------------------------------------------------------------------
SECURE_SSL_REDIRECT = config("SECURE_SSL_REDIRECT", default=False, cast=bool)  # FIXED: False for dev
SESSION_COOKIE_SECURE = config("SESSION_COOKIE_SECURE", default=False, cast=bool)
CSRF_COOKIE_SECURE = config("CSRF_COOKIE_SECURE", default=False, cast=bool)
SECURE_HSTS_SECONDS = config("SECURE_HSTS_SECONDS", default=0, cast=int)  # FIXED: 0 for dev
SECURE_HSTS_INCLUDE_SUBDOMAINS = config("SECURE_HSTS_INCLUDE_SUBDOMAINS", default=False, cast=bool)
SECURE_HSTS_PRELOAD = config("SECURE_HSTS_PRELOAD", default=False, cast=bool)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = config(
    "SECURE_REFERRER_POLICY", default="no-referrer-when-downgrade"
)
X_FRAME_OPTIONS = "DENY"
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_HTTPONLY = True

# ----------------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------------
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = config("CORS_ALLOWED_ORIGINS", default="http://localhost:3000", cast=Csv())
CORS_URLS_REGEX = config("CORS_URLS_REGEX", default=r"^/api/.*$")
CORS_ALLOW_HEADERS = [*list(default_headers), "Authorization", "X-Requested-With"]

# ----------------------------------------------------------------------------
# DRF / JWT / OpenAPI
# ----------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",  # FIXED: core.pagination may not exist
    "PAGE_SIZE": 20,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

# ----------------------------------------------------------------------------
# OpenAPI/Swagger Configuration (drf-spectacular)
# ----------------------------------------------------------------------------
SPECTACULAR_SETTINGS = {
    "TITLE": "Hydraulic Diagnostic SaaS API",
    "DESCRIPTION": "Enterprise hydraulic systems diagnostic platform with AI-powered analysis",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "SCHEMA_PATH_PREFIX": "/api/",
    "SERVERS": [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.hydraulic-diagnostic.com", "description": "Production server"},
    ],
}

# ----------------------------------------------------------------------------
# TimescaleDB Settings
# ----------------------------------------------------------------------------
TIMESCALE_ENABLED = config("TIMESCALE_ENABLED", default=False, cast=bool)

# ----------------------------------------------------------------------------
# Celery Configuration
# ----------------------------------------------------------------------------
CELERY_BROKER_URL = config("REDIS_URL", default="redis://redis:6379/0")
CELERY_RESULT_BACKEND = config("REDIS_URL", default="redis://redis:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_RESULT_SERIALIZER = "json"
CELERY_TASK_SERIALIZER = "json"
TIME_ZONE = "UTC"
CELERY_TIMEZONE = TIME_ZONE
CELERY_ENABLE_UTC = True

# ----------------------------------------------------------------------------
# i18n / tz
# ----------------------------------------------------------------------------
LANGUAGE_CODE = "ru-ru"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTH_USER_MODEL = "users.User"

# ----------------------------------------------------------------------------
# Logging (structured)
# ----------------------------------------------------------------------------
DJANGO_LOG_LEVEL = os.getenv("DJANGO_LOG_LEVEL", "INFO")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "structlog.stdlib.ProcessorFormatter",
            "processor": structlog.processors.JSONRenderer(),
        },
        "console": {
            "()": "structlog.stdlib.ProcessorFormatter",
            "processor": structlog.dev.ConsoleRenderer(),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if not DEBUG else "console",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {"level": DJANGO_LOG_LEVEL},
        "django.request": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "django.db.backends": {"handlers": ["console"], "level": "ERROR"},
        "celery": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# ----------------------------------------------------------------------------
# ML SERVICE CONFIGURATION
# ----------------------------------------------------------------------------
ML_SERVICE_URL = config("ML_SERVICE_URL", default="http://localhost:8001", cast=str)
ML_SERVICE_TIMEOUT = 5.0
ML_SERVICE_RETRY_ATTEMPTS = 3
