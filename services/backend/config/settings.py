"""Django settings for Hydraulic Diagnostics Backend.

Production-ready configuration with:
- Python 3.14+ support
- ASGI/async middleware
- Environment-based configuration
- Security hardening

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
    "apps.core",  # Shared enums, utilities
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
    "whitenoise.middleware.WhiteNoiseMiddleware",  # Static files
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "apps.core.middleware.RateLimitMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Custom async middleware (comment out for sync-only deployment)
    # "apps.core.middleware.AsyncRequestLoggingMiddleware",
    # "apps.monitoring.middleware.AsyncRequestLoggingMiddleware",
    # "django_prometheus.middleware.PrometheusAfterMiddleware",
]

ROOT_URLCONF = "config.urls"

# ============================================================
# ASGI/WSGI CONFIGURATION
# ============================================================
# For async support (Daphne, Uvicorn, Hypercorn)
ASGI_APPLICATION = "config.asgi.application"

# For sync-only deployment (Gunicorn with sync workers)
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
            ],
        },
    },
]

# ============================================================
# Use SQLite for development, PostgreSQL for production
# ============================================================
if DEBUG:
    # Development: SQLite (no setup needed)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
    print("🔧 Development mode: Using SQLite database")
else:
    # ============================================================
    # DATABASE (PostgreSQL)
    # ============================================================
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("DATABASE_NAME", "hydraulic_db"),
            "USER": os.getenv("DATABASE_USER", "postgres"),
            "PASSWORD": os.getenv("DATABASE_PASSWORD", "postgres"),
            "HOST": os.getenv("DATABASE_HOST", "postgres"),
            "PORT": os.getenv("DATABASE_PORT", "5432"),
            "CONN_MAX_AGE": 600,  # Connection pooling
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

# Use Argon2 for password hashing (most secure)
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
]

# ============================================================
# INTERNATIONALIZATION
# ============================================================

LANGUAGE_CODE = "ru-ru"  # Было: "en-us"
TIME_ZONE = "Europe/Moscow"  # Твой часовой пояс
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Поддерживаемые языки
LANGUAGES = [
    ("ru", "Русский"),
    ("en", "English"),
]

# Путь к файлам переводов
LOCALE_PATHS = [
    BASE_DIR / "locale",
]


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
# STATIC FILES (CSS, JavaScript, Images)
# ============================================================

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_DIRS = [
    BASE_DIR / "static",  # Дополнительная папка для твоих файлов
]

STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]

# Whitenoise для статики (уже должно быть)
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Media files
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
CELERY_TASK_SOFT_TIME_LIMIT = 300  # 5 minutes
CELERY_TASK_TIME_LIMIT = 360  # 6 minutes (hard limit)
CELERY_WORKER_LOG_FORMAT = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
CELERY_WORKER_TASK_LOG_FORMAT = (
    "[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s"
)

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
SERVER_EMAIL = DEFAULT_FROM_EMAIL  # FRONTEND_URL для email-ссылок (если еще нет)
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
# ADMIN CUSTOMIZATION
# ============================================================
ADMIN_SITE_HEADER = "Hydraulic Diagnostics - Admin Panel"
ADMIN_SITE_TITLE = "HDX Admin"
ADMIN_INDEX_TITLE = "Customer Support & Operations"

# ============================================================
# SECURITY SETTINGS (Production)
# ============================================================
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_SECONDS = 31536000  # 1 year
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
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "filters": {
        "require_debug_false": {
            "()": "django.utils.log.RequireDebugFalse",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "mail_admins": {
            "level": "ERROR",
            "class": "django.utils.log.AdminEmailHandler",
            "filters": ["require_debug_false"],
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
        "django.request": {
            "handlers": ["console", "mail_admins"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}

# ============================================================
# DEFAULT PRIMARY KEY FIELD TYPE
# ============================================================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ============================================================
# SENTRY - Error tracking (only in production)
# ============================================================
SENTRY_DSN = os.getenv("SENTRY_DSN", "")

# Only initialize Sentry in production with valid DSN
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
    if DEBUG:
        print("🆎  Sentry disabled in development mode")
    else:
        print("⚠️  Sentry DSN not configured")
