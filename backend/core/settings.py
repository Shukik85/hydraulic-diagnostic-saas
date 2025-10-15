from datetime import timedelta
import os
import sys
from pathlib import Path
from decouple import config
import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

# Security
SECRET_KEY = config(
    'SECRET_KEY', default='django-insecure-change-in-production-key-12345')
DEBUG = config('DEBUG', default=True, cast=bool)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1',
                       cast=lambda v: [s.strip() for s in v.split(',')])

# Application definition
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

THIRD_PARTY_APPS = [
    'rest_framework',
    'rest_framework_simplejwt',  # JWT authentication
    'corsheaders',  # CORS headers
    'django_filters',  # Django filters для фильтрации API
]

LOCAL_APPS = [
    'apps.users.apps.UsersConfig',
    'apps.diagnostics.apps.DiagnosticsConfig',
    'apps.rag_assistant.apps.RagAssistantConfig',
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS должен быть первым
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'

# Database
DATABASES = {
    'default': dj_database_url.parse(
        config('DATABASE_URL',
               default='postgresql://postgres:password123@localhost:5432/hydraulic_diagnostic')
    )
}

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',  # JWT authentication
    ),
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',  # По умолчанию требуется аутентификация
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',  # Django filters
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',  # Базовая пагинация
    'PAGE_SIZE': 20,  # Размер страницы по умолчанию
}

# JWT Configuration
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),  # Access token живет 1 час
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),  # Refresh token живет 7 дней
    'ROTATE_REFRESH_TOKENS': True,  # Обновлять refresh token при использовании
    'BLACKLIST_AFTER_ROTATION': True,  # Добавлять старые токены в blacklist
    'UPDATE_LAST_LOGIN': True,  # Обновлять last_login при аутентификации
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'TOKEN_TYPE_CLAIM': 'token_type',
}

# CORS Settings - настройки для кросс-доменных запросов
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True  # Разрешить отправку cookies

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# Internationalization
LANGUAGE_CODE = 'ru-ru'
TIME_ZONE = 'Europe/Moscow'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom user model
AUTH_USER_MODEL = 'users.User'

# AI и Machine Learning настройки
AI_SETTINGS = {
    'ENABLE_AI_ANALYSIS': True,
    'AI_MODEL_PATH': os.path.join(BASE_DIR, 'models'),
    'RAG_DATABASE_PATH': os.path.join(BASE_DIR, 'knowledge_base.db'),
    'MAX_SENSOR_DATA_BATCH_SIZE': 1000,
    'AI_ANALYSIS_CACHE_TIMEOUT': 3600,  # 1 час
}

# Настройки RAG системы
RAG_SETTINGS = {
    'KNOWLEDGE_BASE_MAX_DOCUMENTS': 10000,
    'VECTOR_DIMENSIONS': 1000,
    'SIMILARITY_THRESHOLD': 0.1,
    'MAX_SEARCH_RESULTS': 20,
}

# Мониторинг и алерты
MONITORING_SETTINGS = {
    'ENABLE_AUTO_ALERTS': True,
    'CRITICAL_EVENTS_THRESHOLD': 5,  # События за час
    'AUTO_REPORT_GENERATION': True,
    'HEALTH_CHECK_INTERVAL': 300,  # 5 минут
}

# Создание папки для логов
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} | {name} | {funcName}:{lineno} | {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '[{levelname}] {asctime} | {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
        },
    },
    'handlers': {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": sys.stdout,
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'django.log'),
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            "encoding": "utf-8",
            'formatter': 'verbose',
            'level': 'DEBUG',
        },
        'diagnostic_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'diagnostic.log'),
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
            'level': 'DEBUG',
        },
        'ai_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'ai_engine.log'),
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 3,
            'formatter': 'verbose',
            'level': 'DEBUG',
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'errors.log'),
            'maxBytes': 1024*1024*5,  # 5MB
            'backupCount': 10,
            'formatter': 'verbose',
            'level': 'ERROR',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.server': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'apps.diagnostics': {
            'handlers': ['console', 'diagnostic_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'apps.rag_assistant': {
            'handlers': ['console', 'ai_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'error_file'],
        'level': 'WARNING',
    },
}

# Кеширование для AI операций
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'diagnostic-cache',
        'TIMEOUT': 3600,
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}

# Настройки безопасности для production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# Максимальный размер загружаемых файлов (100MB)
FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
