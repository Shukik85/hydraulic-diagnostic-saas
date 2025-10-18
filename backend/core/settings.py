import os
import sys
from datetime import timedelta
from pathlib import Path

import dj_database_url
from decouple import Csv, config

BASE_DIR = Path(__file__).resolve().parent.parent

# Security
SECRET_KEY = config("SECRET_KEY", default="django-insecure-change-in-production-key-12345")
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="localhost,127.0.0.1", cast=Csv())

# Application definition
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
    "rest_framework_simplejwt",  # JWT authentication
    "corsheaders",  # CORS headers
    "django_filters",  # Django filters для фильтрации API
    "django_ratelimit",  # Rate limiting
]

LOCAL_APPS = [
    "apps.users.apps.UsersConfig",
    "apps.diagnostics.apps.DiagnosticsConfig",
    "apps.rag_assistant.apps.RagAssistantConfig",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# ОПТИМИЗИРОВАННЫЙ MIDDLEWARE - ПОРЯДОК ВАЖЕН!
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
    "apps.rag_assistant.middleware.PerformanceMonitoringMiddleware",  # Новый!
    "django.middleware.cache.FetchFromCacheMiddleware",  # Последним!
]

# Для django-debug-toolbar (в development):
if DEBUG:
    INSTALLED_APPS += ["debug_toolbar"]
    MIDDLEWARE.insert(-2, "debug_toolbar.middleware.DebugToolbarMiddleware")  # Перед Performance
    INTERNAL_IPS = ["127.0.0.1", "localhost"]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
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

WSGI_APPLICATION = "core.wsgi.application"

# Database - ОПТИМИЗИРОВАННОЕ ПОДКЛЮЧЕНИЕ
DATABASES = {
    "default": dj_database_url.parse(
        config(
            "DATABASE_URL",
            default="postgresql://postgres:password123@localhost:5432/hydraulic_diagnostic",
        )
    )
}

# Оптимизация соединений с БД
DATABASES["default"]["CONN_MAX_AGE"] = config("DATABASE_CONN_MAX_AGE", default=600, cast=int)
DATABASES["default"]["CONN_HEALTH_CHECKS"] = True

# Redis Configuration
REDIS_URL = config("REDIS_URL", default="redis://localhost:6379/0")
AI_REDIS_URL = config("AI_REDIS_URL", default="redis://localhost:6379/1")

# ОПТИМИЗИРОВАННОЕ КЕШИРОВАНИЕ
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
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
        "LOCATION": AI_REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {"max_connections": 20},
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
        },
        "KEY_PREFIX": "ai_operations",
        "TIMEOUT": config("AI_CACHE_TIMEOUT", default=86400, cast=int),  # 24 часа для AI
    },
}

# Кеширование страниц
CACHE_MIDDLEWARE_ALIAS = "default"
CACHE_MIDDLEWARE_SECONDS = 300  # 5 минут
CACHE_MIDDLEWARE_KEY_PREFIX = "hydraulic"

# Session кеширование
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
SESSION_COOKIE_AGE = 86400  # 24 часа
SESSION_SAVE_EVERY_REQUEST = False  # Оптимизация

# ОПТИМИЗИРОВАННЫЕ REST Framework настройки
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",  # JWT authentication
    ),
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",  # По умолчанию требуется аутентификация
    ],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",  # Django filters
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardResultsSetPagination",  # ОПТИМИЗИРОВАННО
    "PAGE_SIZE": 20,  # Размер страницы по умолчанию
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ]
    + (["rest_framework.renderers.BrowsableAPIRenderer"] if DEBUG else []),
    # Rate limiting - базовые настройки
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/day",
        "user": "1000/day",
    },
}

# JWT Configuration
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),  # Access token живет 1 час
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),  # Refresh token живет 7 дней
    "ROTATE_REFRESH_TOKENS": True,  # Обновлять refresh token при использовании
    "BLACKLIST_AFTER_ROTATION": True,  # Добавлять старые токены в blacklist
    "UPDATE_LAST_LOGIN": True,  # Обновлять last_login при аутентификации
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "AUTH_HEADER_TYPES": ("Bearer",),
    "AUTH_HEADER_NAME": "HTTP_AUTHORIZATION",
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    "TOKEN_TYPE_CLAIM": "token_type",
}

# CORS Settings - настройки для кросс-доменных запросов
CORS_ALLOWED_ORIGINS = config(
    "CORS_ALLOWED_ORIGINS",
    default="http://localhost:3000,http://127.0.0.1:3000",
    cast=Csv(),
)

CSRF_TRUSTED_ORIGINS = config(
    "CSRF_TRUSTED_ORIGINS",
    default="http://localhost:3000,http://127.0.0.1:3000",
    cast=Csv(),
)

CORS_ALLOW_CREDENTIALS = True  # Разрешить отправку cookies

CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

# Internationalization
LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True

# Static files - ОПТИМИЗИРОВАННОЕ
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Custom user model
AUTH_USER_MODEL = "users.User"

# AI и Machine Learning настройки - ОПТИМИЗИРОВАННЫЕ
AI_SETTINGS = {
    "ENABLE_AI_ANALYSIS": True,
    "AI_MODEL_PATH": os.path.join(BASE_DIR, "models"),
    "RAG_DATABASE_PATH": os.path.join(BASE_DIR, "knowledge_base.db"),
    "MAX_SENSOR_DATA_BATCH_SIZE": 1000,
    "AI_ANALYSIS_CACHE_TIMEOUT": 3600,  # 1 час
    "OPENAI_API_KEY": config("OPENAI_API_KEY", default=""),
    # ОПТИМИЗАЦИИ
    "ENABLE_CACHING": True,
    "CACHE_EMBEDDINGS": True,
    "CACHE_SEARCH_RESULTS": True,
    "BATCH_PROCESSING": True,
    "MAX_CONCURRENT_REQUESTS": 5,
    "REQUEST_TIMEOUT": 30,
    "ENABLE_COMPRESSION": True,
    "MAX_QUERY_LENGTH": 500,
    "MAX_CONTENT_SIZE": 50 * 1024 * 1024,  # 50MB
    "RATE_LIMIT_AI_REQUESTS": True,
    "SANITIZE_INPUT": True,
}

# Настройки RAG системы
RAG_SETTINGS = {
    "KNOWLEDGE_BASE_MAX_DOCUMENTS": 10000,
    "VECTOR_DIMENSIONS": 1000,
    "SIMILARITY_THRESHOLD": 0.1,
    "MAX_SEARCH_RESULTS": 20,
}

# Мониторинг и алерты - ОПТИМИЗИРОВАННОЕ
MONITORING_SETTINGS = {
    "ENABLE_AUTO_ALERTS": True,
    "CRITICAL_EVENTS_THRESHOLD": 5,  # События за час
    "AUTO_REPORT_GENERATION": True,
    "HEALTH_CHECK_INTERVAL": 300,  # 5 минут
}

# Performance thresholds
SLOW_REQUEST_THRESHOLD = 1.0  # 1 секунда
VERY_SLOW_THRESHOLD = 5.0  # 5 секунд

# Создание папки для логов
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ОПТИМИЗИРОВАННОЕ STRUCTURED LOGGING
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} | {name} | {funcName}:{lineno} | {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "[{levelname}] {asctime} | {message}",
            "style": "{",
            "datefmt": "%H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": sys.stdout,
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "django.log"),
            "maxBytes": 1024 * 1024 * 10,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
            "formatter": "verbose",
            "level": "DEBUG",
        },
        "diagnostic_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "diagnostic.log"),
            "maxBytes": 1024 * 1024 * 10,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
            "level": "DEBUG",
        },
        "ai_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "ai_engine.log"),
            "maxBytes": 1024 * 1024 * 10,  # 10MB
            "backupCount": 3,
            "formatter": "verbose",
            "level": "DEBUG",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "errors.log"),
            "maxBytes": 1024 * 1024 * 5,  # 5MB
            "backupCount": 10,
            "formatter": "verbose",
            "level": "ERROR",
        },
        "slow_query_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "slow_queries.log"),
            "maxBytes": 1024 * 1024 * 5,  # 5MB
            "backupCount": 5,
            "formatter": "verbose",
            "level": "WARNING",
        },
        "performance": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOGS_DIR / "performance.log"),
            "maxBytes": 1024 * 1024 * 20,  # 20MB
            "backupCount": 5,
            "formatter": "json",  # JSON для structured logging
            "level": "INFO",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "django.server": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "django.db.backends": {
            "handlers": ["slow_query_file"],
            "level": "DEBUG" if DEBUG else "WARNING",
            "propagate": False,
        },
        "apps": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "apps.diagnostics": {
            "handlers": ["console", "diagnostic_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "apps.rag_assistant": {
            "handlers": ["console", "ai_file", "slow_query_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "performance": {
            "handlers": ["performance"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "error_file"],
        "level": "WARNING",
    },
}

# Celery Configuration - ОПТИМИЗИРОВАННОЕ
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 минут

# Оптимизация Celery
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_TASK_DEFAULT_RETRY_DELAY = 60
CELERY_TASK_MAX_RETRIES = 3

# Максимальный размер загружаемых файлов - ОПТИМИЗИРОВАННО
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
FILE_UPLOAD_PERMISSIONS = 0o644

# Оптимизация File Upload
FILE_UPLOAD_HANDLERS = [
    "django.core.files.uploadhandler.MemoryFileUploadHandler",
    "django.core.files.uploadhandler.TemporaryFileUploadHandler",
]

# Version
VERSION = "1.0.0"

# ПОДКЛЮЧЕНИЕ ОПТИМИЗАЦИЙ В PRODUCTION
if not DEBUG:
    # Импортируем оптимизации и безопасность
    try:
        pass

        print("⚡ Production optimizations loaded")
    except ImportError:
        pass

print(f"🚀 Django settings loaded - DEBUG: {DEBUG}, VERSION: {VERSION}")
