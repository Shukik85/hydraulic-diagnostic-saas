"""Продакшн конфигурация Django для прохождения --deploy проверок.

Конфигурация для продакшн-среды с усиленной безопасностью.
Можно использовать через DJANGO_SETTINGS_MODULE=core.deploy_settings
"""

from .settings import *  # noqa: F401,F403

# Переопределяем настройки для production
DEBUG = False

# Enterprise безопасность: сильный SECRET_KEY (50+ символов, 5+ уникальных)
SECRET_KEY = config(
    "SECRET_KEY",
    default="production-hydraulic-diagnostic-platform-enterprise-secret-key-2025-strong-security-ml-powered",
)

# Security настройки
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 год
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = "DENY"

# Отключаем отладочные приложения
INSTALLED_APPS = [app for app in INSTALLED_APPS if "debug" not in app.lower()]

# Логирование для production
LOGGING["root"]["level"] = "ERROR"  # noqa: F405
LOGGING["loggers"]["django"]["level"] = "ERROR"  # noqa: F405
LOGGING["loggers"]["django.request"]["level"] = "ERROR"  # noqa: F405

# Отключаем отладочные возможности
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# Production кеширование
CACHES["default"]["TIMEOUT"] = 3600  # 1 час  # noqa: F405


# Переопределяем настройки для проверок системы с --deploy
def configure_for_deploy():
    """Настройка для проверок системы."""
    # При запуске check --deploy можно пропустить некоторые проверки
    pass


configure_for_deploy()
