"""Модуль проекта с автогенерированным докстрингом."""

# backend/core/__init__.py
import os
import sys

import django

from .celery import app as celery_app


# Автоподключение Django в интерактивных сессиях (python/ipython),
# чтобы можно было сразу делать импорты из проекта, не вызывая вручную django.setup().
def _auto_setup_django_for_shell() -> None:
    # Срабатывает только в интерактивных сессиях
    if not hasattr(sys, "ps1"):
        return

    # Если уже настроено окружение — ничего не делаем
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return

    # Настройка окружения и инициализация Django
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

        django.setup()
        print("✅ Django auto-setup: core.settings")
    except Exception as exc:
        print(f"⚠️ Django auto-setup failed: {exc}")


_auto_setup_django_for_shell()


__all__ = ("celery_app",)
