"""Модуль проекта с автогенерированным докстрингом."""

# core/secure_settings.py
# КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ БЕЗОПАСНОСТИ (typed)
from __future__ import annotations

from typing import Any

from decouple import Csv, config

from . import settings as base

# Базовые настройки
SECRET_KEY: str = config("SECRET_KEY")
DEBUG: bool = False
ALLOWED_HOSTS = config("ALLOWED_HOSTS", cast=Csv())

# Security Headers - КРИТИЧЕСКИЕ для production
SECURE_BROWSER_XSS_FILTER: bool = True
SECURE_CONTENT_TYPE_NOSNIFF: bool = True
X_FRAME_OPTIONS: str = "DENY"
SECURE_HSTS_SECONDS: int = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS: bool = True
SECURE_HSTS_PRELOAD: bool = True
SECURE_SSL_REDIRECT: bool = True
CSRF_COOKIE_SECURE: bool = True
SESSION_COOKIE_SECURE: bool = True
SECURE_REFERRER_POLICY: str = "strict-origin-when-cross-origin"

# Session Security
SESSION_COOKIE_HTTPONLY: bool = True
CSRF_COOKIE_HTTPONLY: bool = True
SESSION_EXPIRE_AT_BROWSER_CLOSE: bool = True
SESSION_COOKIE_AGE: int = 3600

# Rate Limiting для API - КРИТИЧНО для защиты от атак
REST_FRAMEWORK: dict[str, Any] = (
    base.REST_FRAMEWORK.copy() if hasattr(base, "REST_FRAMEWORK") else {}
)
REST_FRAMEWORK.update(
    {
        "DEFAULT_THROTTLE_CLASSES": [
            "rest_framework.throttling.AnonRateThrottle",
            "rest_framework.throttling.UserRateThrottle",
        ],
        "DEFAULT_THROTTLE_RATES": {
            "anon": "100/day",
            "user": "1000/day",
            "ai_queries": "50/hour",
            "file_upload": "10/hour",
            "login": "5/minute",
        },
    }
)

# CORS Security - строгие настройки
CORS_ALLOWED_ORIGINS = config("CORS_ALLOWED_ORIGINS", cast=Csv(), default="")
CORS_ALLOWED_ORIGINS = [o for o in CORS_ALLOWED_ORIGINS if o]
CORS_ALLOW_CREDENTIALS: bool = False
CSRF_TRUSTED_ORIGINS = config("CSRF_TRUSTED_ORIGINS", cast=Csv(), default="")
CSRF_TRUSTED_ORIGINS = [o for o in CSRF_TRUSTED_ORIGINS if o]

# Дополнительные заголовки безопасности
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_TZ: bool = True

# File Upload Security
FILE_UPLOAD_MAX_MEMORY_SIZE: int = 10 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE: int = 10 * 1024 * 1024
FILE_UPLOAD_PERMISSIONS: int = 0o644

# ДОПУСТИМЫЕ типы файлов - только безопасные
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "md"]
MAX_FILE_SIZE: int = 10 * 1024 * 1024

# Logging Security Events
LOGGING: dict[str, Any] = base.LOGGING.copy() if hasattr(base, "LOGGING") else {}
LOGGING.setdefault("loggers", {})
LOGGING["loggers"]["security"] = {
    "handlers": ["file", "error_file"],
    "level": "WARNING",
    "propagate": False,
}

# AI Settings Security
AI_SETTINGS: dict[str, Any] = (
    base.AI_SETTINGS.copy() if hasattr(base, "AI_SETTINGS") else {}
)
AI_SETTINGS.update(
    {
        "MAX_QUERY_LENGTH": 500,
        "MAX_CONTENT_SIZE": 10 * 1024 * 1024,
        "RATE_LIMIT_AI_REQUESTS": True,
        "SANITIZE_INPUT": True,
    }
)

# Database Security
DATABASES: dict[str, Any] = base.DATABASES.copy() if hasattr(base, "DATABASES") else {}
if "default" in DATABASES:
    default_db: dict[str, Any] = DATABASES["default"]
    default_db.setdefault("OPTIONS", {})
    options: dict[str, Any] = default_db["OPTIONS"]
    options.update(
        {
            "sslmode": "require",
            "options": "-c default_transaction_isolation=serializable",
        }
    )

print("🔒 SECURE SETTINGS LOADED - Production Security Enabled")
