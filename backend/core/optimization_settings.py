# core/optimization_settings.py (typed)
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from decouple import config

from . import settings as base

# Database Connection Pooling - КРИТИЧНО для production
DATABASES: Dict[str, Any] = base.DATABASES.copy() if hasattr(base, "DATABASES") else {}
if "default" in DATABASES:
    default_db: Dict[str, Any] = DATABASES["default"]
    default_db.update(
        {
            "CONN_MAX_AGE": config("DATABASE_CONN_MAX_AGE", default=600, cast=int),
            "CONN_HEALTH_CHECKS": config("DATABASE_CONN_HEALTH_CHECKS", default=True, cast=bool),
        }
    )

# Мониторинг медленных запросов - ОЧЕНЬ ВАЖНО
LOGGING: Dict[str, Any] = base.LOGGING.copy() if hasattr(base, "LOGGING") else {}
LOGGING.setdefault("loggers", {})
LOGGING.setdefault("handlers", {})
LOGGING["loggers"]["django.db.backends"] = {
    "level": "DEBUG" if getattr(base, "DEBUG", False) else "WARNING",
    "handlers": ["slow_query_file"],
    "propagate": False,
}

# Улучшенное кеширование - КЛЮЧЕВАЯ оптимизация
CACHES: Dict[str, Any] = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": config("REDIS_URL", default="redis://localhost:6379/0"),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 50,
                "retry_on_timeout": True,
            },
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
            "IGNORE_EXCEPTIONS": True,
        },
        "KEY_PREFIX": "hydraulic_diagnostic",
        "VERSION": 1,
        "TIMEOUT": config("CACHE_TIMEOUT", default=3600, cast=int),
    },
    "ai_cache": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": config("AI_REDIS_URL", default=config("REDIS_URL", default="redis://localhost:6379/0")),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {"max_connections": 20},
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
        },
        "KEY_PREFIX": "ai_operations",
        "TIMEOUT": config("AI_CACHE_TIMEOUT", default=86400, cast=int),
    },
}

# Session кеширование - повышает производительность
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
SESSION_COOKIE_AGE: int = 86400
SESSION_SAVE_EVERY_REQUEST: bool = False

# Middleware оптимизация - порядок важен!
MIDDLEWARE: List[str] = [
    "django.middleware.cache.UpdateCacheMiddleware",
    "django.middleware.gzip.GZipMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.rag_assistant.middleware.PerformanceMonitoringMiddleware",
    "django.middleware.cache.FetchFromCacheMiddleware",
]

# Кеширование страниц
CACHE_MIDDLEWARE_ALIAS = "default"
CACHE_MIDDLEWARE_SECONDS: int = 300
CACHE_MIDDLEWARE_KEY_PREFIX = "hydraulic"

# Static Files оптимизация
STORAGES: Dict[str, Any] = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

# Улучшенные настройки REST Framework
REST_FRAMEWORK: Dict[str, Any] = base.REST_FRAMEWORK.copy() if hasattr(base, "REST_FRAMEWORK") else {}
REST_FRAMEWORK.update(
    {
        "DEFAULT_RENDERER_CLASSES": [
            "rest_framework.renderers.JSONRenderer",
        ],
        "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardResultsSetPagination",
        "PAGE_SIZE": 20,
        "DEFAULT_FILTER_BACKENDS": [
            "django_filters.rest_framework.DjangoFilterBackend",
            "rest_framework.filters.SearchFilter",
            "rest_framework.filters.OrderingFilter",
        ],
        "DEFAULT_CONTENT_NEGOTIATION_CLASS": "rest_framework.content_negotiation.DefaultContentNegotiation",
    }
)

# Оптимизация AI настроек
AI_SETTINGS: Dict[str, Any] = base.AI_SETTINGS.copy() if hasattr(base, "AI_SETTINGS") else {}
AI_SETTINGS.update(
    {
        "ENABLE_CACHING": True,
        "CACHE_EMBEDDINGS": True,
        "CACHE_SEARCH_RESULTS": True,
        "BATCH_PROCESSING": True,
        "MAX_CONCURRENT_REQUESTS": 5,
        "REQUEST_TIMEOUT": 30,
        "ENABLE_COMPRESSION": True,
    }
)

# File Upload оптимизация
FILE_UPLOAD_HANDLERS: List[str] = [
    "django.core.files.uploadhandler.MemoryFileUploadHandler",
    "django.core.files.uploadhandler.TemporaryFileUploadHandler",
]

# Template оптимизация
TEMPLATES = base.TEMPLATES.copy() if hasattr(base, "TEMPLATES") else []
if TEMPLATES:
    TEMPLATES[0] = TEMPLATES[0].copy()
    options_obj = TEMPLATES[0].get("OPTIONS", {})
    options: Dict[str, Any] = dict(options_obj) if isinstance(options_obj, dict) else {}

    loaders_cached: Tuple[str, List[str]] = (
        "django.template.loaders.cached.Loader",
        [
            "django.template.loaders.filesystem.Loader",
            "django.template.loaders.app_directories.Loader",
        ],
    )
    loaders_uncached: List[str] = [
        "django.template.loaders.filesystem.Loader",
        "django.template.loaders.app_directories.Loader",
    ]

    options.update(
        {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
            "loaders": ([loaders_cached] if not getattr(base, "DEBUG", False) else loaders_uncached),
        }
    )
    TEMPLATES[0]["OPTIONS"] = options

# Оптимизация логирования
LOGS_DIR = getattr(base, "LOGS_DIR", None)
if LOGS_DIR is not None:
    LOGGING.setdefault("handlers", {})
    LOGGING["handlers"]["performance"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(LOGS_DIR / "performance.log"),
        "maxBytes": 1024 * 1024 * 20,
        "backupCount": 5,
        "formatter": "verbose",
        "level": "INFO",
    }
    LOGGING.setdefault("loggers", {})
    LOGGING["loggers"]["performance"] = {
        "handlers": ["performance"],
        "level": "INFO",
        "propagate": False,
    }

# Email оптимизация - async email sending
if not getattr(base, "DEBUG", False):
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
    EMAIL_HOST = config("EMAIL_HOST", default="localhost")
    EMAIL_PORT = config("EMAIL_PORT", default=587, cast=int)
    EMAIL_USE_TLS = config("EMAIL_USE_TLS", default=True, cast=bool)
    EMAIL_HOST_USER = config("EMAIL_HOST_USER", default="")
    EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")

print("⚡ PERFORMANCE OPTIMIZATION LOADED - Enhanced Speed & Caching Enabled")
