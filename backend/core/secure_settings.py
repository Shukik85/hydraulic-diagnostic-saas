# core/secure_settings.py
# КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ БЕЗОПАСНОСТИ

from decouple import Csv, config

from .settings import *

# КРИТИЧНО: Никогда не используйте default SECRET_KEY в production
SECRET_KEY = config("SECRET_KEY")  # Обязательно через environment variable
DEBUG = False
ALLOWED_HOSTS = config("ALLOWED_HOSTS", cast=Csv())

# Security Headers - КРИТИЧЕСКИЕ для production
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_HSTS_SECONDS = 31536000  # 1 год
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_SSL_REDIRECT = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"

# Session Security
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_AGE = 3600  # 1 час

# Rate Limiting для API - КРИТИЧНО для защиты от атак
REST_FRAMEWORK["DEFAULT_THROTTLE_CLASSES"] = [
    "rest_framework.throttling.AnonRateThrottle",
    "rest_framework.throttling.UserRateThrottle",
]

REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"] = {
    "anon": "100/day",  # Анонимные пользователи
    "user": "1000/day",  # Авторизованные пользователи
    "ai_queries": "50/hour",  # AI операции - особо лимитированы
    "file_upload": "10/hour",  # Загрузка файлов
    "login": "5/minute",  # Попытки входа
}

# CORS Security - строгие настройки
CORS_ALLOWED_ORIGINS = config("CORS_ALLOWED_ORIGINS", cast=Csv())
CORS_ALLOW_CREDENTIALS = False  # Изменено с True на False для безопасности
CSRF_TRUSTED_ORIGINS = config("CSRF_TRUSTED_ORIGINS", cast=Csv())

# Дополнительные заголовки безопасности
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_TZ = True

# File Upload Security
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB вместо 100MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_PERMISSIONS = 0o644

# ДОПУСТИМЫЕ типы файлов - только безопасные
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "md"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Logging Security Events
LOGGING["loggers"]["security"] = {
    "handlers": ["file", "error_file"],
    "level": "WARNING",
    "propagate": False,
}

# AI Settings Security
AI_SETTINGS.update(
    {
        "MAX_QUERY_LENGTH": 500,
        "MAX_CONTENT_SIZE": 10 * 1024 * 1024,  # 10MB
        "RATE_LIMIT_AI_REQUESTS": True,
        "SANITIZE_INPUT": True,
    }
)

# Database Security
DATABASES["default"]["OPTIONS"] = {
    "sslmode": "require",
    "options": "-c default_transaction_isolation=serializable",
}

print("🔒 SECURE SETTINGS LOADED - Production Security Enabled")
