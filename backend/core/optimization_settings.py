# core/optimization_settings.py
# ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ


from decouple import config

from .settings import *

# Database Connection Pooling - КРИТИЧНО для production
DATABASES["default"].update(
    {
        "CONN_MAX_AGE": config("DATABASE_CONN_MAX_AGE", default=600, cast=int),  # 10 минут
        "CONN_HEALTH_CHECKS": config("DATABASE_CONN_HEALTH_CHECKS", default=True, cast=bool),
        "OPTIONS": {
            "MAX_CONNS": 20,
            "MIN_CONNS": 5,
            # PostgreSQL оптимизация
            "OPTIONS": {
                "MAX_CONNS": 20,
            },
        },
    }
)

# Мониторинг медленных запросов - ОЧЕНЬ ВАЖНО
LOGGING["loggers"]["django.db.backends"] = {
    "level": "DEBUG" if DEBUG else "WARNING",
    "handlers": ["slow_query_file"],
    "propagate": False,
}

# Улучшенное кеширование - КЛЮЧЕВАЯ оптимизация
CACHES = {
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
    # Отдельный кеш для AI операций
    "ai_cache": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": config("AI_REDIS_URL", default=REDIS_URL),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {"max_connections": 20},
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
        },
        "KEY_PREFIX": "ai_operations",
        "TIMEOUT": config("AI_CACHE_TIMEOUT", default=86400, cast=int),  # 24 часа для AI
    },
}

# Session кеширование - повышает производительность
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
SESSION_COOKIE_AGE = 86400  # 24 часа
SESSION_SAVE_EVERY_REQUEST = False  # Оптимизация

# Middleware оптимизация - порядок важен!
MIDDLEWARE = [
    "django.middleware.cache.UpdateCacheMiddleware",  # Первым!
    "django.middleware.gzip.GZipMiddleware",  # Сжатие
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.rag_assistant.middleware.PerformanceMonitoringMiddleware",  # Новый middleware!
    "django.middleware.cache.FetchFromCacheMiddleware",  # Последним!
]

# Кеширование страниц
CACHE_MIDDLEWARE_ALIAS = "default"
CACHE_MIDDLEWARE_SECONDS = 300  # 5 минут
CACHE_MIDDLEWARE_KEY_PREFIX = "hydraulic"

# Static Files оптимизация
STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"

# Улучшенные настройки REST Framework
REST_FRAMEWORK.update(
    {
        "DEFAULT_RENDERER_CLASSES": [
            "rest_framework.renderers.JSONRenderer",
            # 'rest_framework.renderers.BrowsableAPIRenderer',  # Отключен для production
        ],
        "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardResultsSetPagination",
        "PAGE_SIZE": 20,
        "DEFAULT_FILTER_BACKENDS": [
            "django_filters.rest_framework.DjangoFilterBackend",
            "rest_framework.filters.SearchFilter",
            "rest_framework.filters.OrderingFilter",
        ],
        # Кеширование API ответов
        "DEFAULT_CONTENT_NEGOTIATION_CLASS": "rest_framework.content_negotiation.DefaultContentNegotiation",
    }
)

# Оптимизация AI настроек
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
FILE_UPLOAD_HANDLERS = [
    "django.core.files.uploadhandler.MemoryFileUploadHandler",
    "django.core.files.uploadhandler.TemporaryFileUploadHandler",
]

# Template оптимизация
TEMPLATES[0]["OPTIONS"].update(
    {
        "context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
        ],
        "loaders": (
            [
                (
                    "django.template.loaders.cached.Loader",
                    [
                        "django.template.loaders.filesystem.Loader",
                        "django.template.loaders.app_directories.Loader",
                    ],
                ),
            ]
            if not DEBUG
            else [
                "django.template.loaders.filesystem.Loader",
                "django.template.loaders.app_directories.Loader",
            ]
        ),
    }
)

# Оптимизация логирования
LOGGING["handlers"]["performance"] = {
    "class": "logging.handlers.RotatingFileHandler",
    "filename": str(LOGS_DIR / "performance.log"),
    "maxBytes": 1024 * 1024 * 20,  # 20MB
    "backupCount": 5,
    "formatter": "verbose",
    "level": "INFO",
}

LOGGING["loggers"]["performance"] = {
    "handlers": ["performance"],
    "level": "INFO",
    "propagate": False,
}

# Email оптимизация - async email sending
if not DEBUG:
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
    EMAIL_HOST = config("EMAIL_HOST", default="localhost")
    EMAIL_PORT = config("EMAIL_PORT", default=587, cast=int)
    EMAIL_USE_TLS = config("EMAIL_USE_TLS", default=True, cast=bool)
    EMAIL_HOST_USER = config("EMAIL_HOST_USER", default="")
    EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")

print("⚡ PERFORMANCE OPTIMIZATION LOADED - Enhanced Speed & Caching Enabled")
