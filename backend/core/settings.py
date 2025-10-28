"""Django settings for hydraulic-diagnostic-saas project.

CI-compatible with TimescaleDB and Celery Beat integration.
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
    "corsheaders",
    "django_filters",
    "django_extensions",
    "django_celery_beat",  # Database-backed periodic tasks for TimescaleDB
]

LOCAL_APPS = [
    "apps.users.apps.UsersConfig",
    "apps.diagnostics.apps.DiagnosticsConfig",
    "apps.rag_assistant.apps.RagAssistantConfig",
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
                "max_size": 20,  # Увеличено для Celery задач
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
SECURE_SSL_REDIRECT = config("SECURE_SSL_REDIRECT", default=True, cast=bool)
SESSION_COOKIE_SECURE = config("SESSION_COOKIE_SECURE", default=True, cast=bool)
CSRF_COOKIE_SECURE = config("CSRF_COOKIE_SECURE", default=True, cast=bool)
SECURE_HSTS_SECONDS = config("SECURE_HSTS_SECONDS", default=31536000, cast=int)
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = config(
    "SECURE_REFERRER_POLICY", default="no-referrer-when-downgrade"
)
X_FRAME_OPTIONS = "DENY"
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_HTTPONLY = True

# (Optional) Content Security Policy — рекомендуется через django-csp
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'",)
CSP_STYLE_SRC = ("'self'", "https:", "'unsafe-inline'")
CSP_IMG_SRC = ("'self'", "data:", "https:")
CSP_CONNECT_SRC = ("'self'",)
CSP_FONT_SRC = ("'self'", "https:", "data:")
CSP_FRAME_ANCESTORS = ("'none'",)

# ----------------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------------
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = config("CORS_ALLOWED_ORIGINS", default="", cast=Csv()) or []
CORS_URLS_REGEX = config("CORS_URLS_REGEX", default=r"^/api/.*$")
CORS_ALLOW_HEADERS = [*list(default_headers), "Authorization", "X-Requested-With"]

# ----------------------------------------------------------------------------
# DRF / JWT
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
    "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardResultsSetPagination",
    "PAGE_SIZE": 20,
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
# TimescaleDB Settings
# ----------------------------------------------------------------------------
TIMESCALE_ENABLED = config("TIMESCALE_ENABLED", default=False, cast=bool)
TIMESCALE_SETTINGS = {
    "DEFAULT_CHUNK_TIME_INTERVAL": config(
        "TIMESCALE_CHUNK_TIME_INTERVAL", default="7 days"
    ),
    "DEFAULT_COMPRESSION_AGE": "30 days",
    "DEFAULT_RETENTION_PERIOD": "1 year",
    "MAINTENANCE_WINDOW_HOUR": 2,  # Час для выполнения обслуживания
    "ENABLE_AUTO_COMPRESSION": True,
    "ENABLE_AUTO_RETENTION": True,
    "HYPERTABLES": [
        {
            "table": "sensor_data",
            "time_column": "timestamp",
            "chunk_time_interval": "7 days",
            "compress_segmentby": ["system_id", "component_id"],
            "compress_orderby": "timestamp DESC",
        }
    ],
}

# ----------------------------------------------------------------------------
# Celery Configuration
# ----------------------------------------------------------------------------
CELERY_BROKER_URL = config("REDIS_URL", default="redis://redis:6379/0")
CELERY_RESULT_BACKEND = config("REDIS_URL", default="redis://redis:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_RESULT_SERIALIZER = "json"
CELERY_TASK_SERIALIZER = "json"
# Fix multiple assignment for flake8 compatibility
TIME_ZONE = "UTC"
CELERY_TIMEZONE = TIME_ZONE
CELERY_ENABLE_UTC = True

# Celery Beat Schedule для TimescaleDB управления
CELERY_BEAT_SCHEDULE = {
    # Ежедневная очистка старых партиций в 2:00 ночи
    "cleanup-old-timescale-partitions": {
        "task": "apps.diagnostics.timescale_tasks.cleanup_old_partitions",
        "schedule": crontab(hour=2, minute=0),  # 02:00 каждый день
        "args": ("sensor_data", "90 days"),
        "options": {
            "queue": "maintenance",
        },
    },
    # Еженедельное сжатие старых chunk'ов по воскресеньям в 3:00
    "compress-old-timescale-chunks": {
        "task": "apps.diagnostics.timescale_tasks.compress_old_chunks",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),  # Воскресенье 03:00
        "args": ("sensor_data", "30 days"),
        "options": {
            "queue": "maintenance",
        },
    },
    # Проверка создания партиций на будущее каждые 6 часов
    "ensure-future-partitions": {
        "task": "apps.diagnostics.timescale_tasks.ensure_partitions_for_range",
        "schedule": timedelta(hours=6),
        "kwargs": {
            "table_name": "sensor_data",
            "start_time": None,  # Текущее время
            "end_time": None,  # +30 дней от текущего времени
            "chunk_interval": "7 days",
        },
        "options": {
            "queue": "maintenance",
        },
    },
    # Сбор статистики по hypertables каждый час
    "collect-hypertable-stats": {
        "task": "apps.diagnostics.timescale_tasks.get_hypertable_stats",
        "schedule": timedelta(hours=1),
        "args": ("sensor_data",),
        "options": {
            "queue": "monitoring",
        },
    },
    # Проверка здоровья TimescaleDB каждые 15 минут
    "timescale-health-check": {
        "task": "apps.diagnostics.timescale_tasks.timescale_health_check",
        "schedule": timedelta(minutes=15),
        "options": {
            "queue": "monitoring",
            "expires": 300,  # Задача истекает через 5 минут
        },
    },
}

# Настройки очередей для разных типов задач
CELERY_TASK_ROUTES = {
    "apps.diagnostics.timescale_tasks.*": {"queue": "timescale"},
    "apps.diagnostics.timescale_tasks.cleanup_old_partitions": {"queue": "maintenance"},
    "apps.diagnostics.timescale_tasks.compress_old_chunks": {"queue": "maintenance"},
    "apps.diagnostics.timescale_tasks.get_hypertable_stats": {"queue": "monitoring"},
}

# Настройки для работы с базой данных в Celery задачах
CELERY_TASK_ANNOTATIONS = {
    "apps.diagnostics.timescale_tasks.cleanup_old_partitions": {
        "rate_limit": "1/m",  # Не более 1 задачи в минуту
        "time_limit": 3600,  # Таймаут 1 час
        "soft_time_limit": 3000,  # Мягкий таймаут 50 минут
    },
    "apps.diagnostics.timescale_tasks.compress_old_chunks": {
        "rate_limit": "1/5m",  # Не более 1 задачи в 5 минут
        "time_limit": 1800,  # Таймаут 30 минут
    },
    "apps.diagnostics.timescale_tasks.ensure_partitions_for_range": {
        "rate_limit": "10/m",  # До 10 задач в минуту
        "time_limit": 300,  # Таймаут 5 минут
    },
}

# Настройки для мониторинга производительности
CELERY_SEND_TASK_EVENTS = True
CELERY_TASK_SEND_SENT_EVENT = True
CELERY_RESULT_EXPIRES = 3600  # Результаты задач хранятся 1 час
CELERY_TASK_RESULT_EXPIRES = 3600

# ----------------------------------------------------------------------------
# i18n / tz
# ----------------------------------------------------------------------------
LANGUAGE_CODE = "ru-ru"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTH_USER_MODEL = "users.User"

# ----------------------------------------------------------------------------
# Logging (structured) with Celery support
# ----------------------------------------------------------------------------

DJANGO_LOG_LEVEL = os.getenv("DJANGO_LOG_LEVEL", "INFO")
DJANGO_REQUEST_LOG_LEVEL = os.getenv("DJANGO_REQUEST_LOG_LEVEL", "WARNING")
DJANGO_DATABASE_LOG_LEVEL = os.getenv("DJANGO_DATABASE_LOG_LEVEL", "ERROR")
CELERY_LOG_LEVEL = os.getenv("CELERY_LOG_LEVEL", "INFO")

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
        "django.request": {
            "handlers": ["console"],
            "level": DJANGO_REQUEST_LOG_LEVEL,
            "propagate": False,
        },
        "django.server": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.db.backends": {
            "handlers": ["console"],
            "level": DJANGO_DATABASE_LOG_LEVEL,
        },
        "celery": {
            "handlers": ["console"],
            "level": CELERY_LOG_LEVEL,
            "propagate": False,
        },
        "celery.task": {
            "handlers": ["console"],
            "level": CELERY_LOG_LEVEL,
            "propagate": False,
        },
        "apps.diagnostics.timescale_tasks": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def _add_request_id(_, __, event_dict):
    """Add request ID to log events."""
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_request_id,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# ----------------------------------------------------------------------------
# Docker deployment notes (docstring-only)
# ----------------------------------------------------------------------------
# WSGI: gunicorn core.wsgi:application -w 4 -b 0.0.0.0:8000
# ASGI: gunicorn core.asgi:application -w 4 -k uvicorn.workers.UvicornWorker
