"""Модуль проекта с автогенерированным докстрингом."""

#!/usr/bin/env python
import os
import sys


def main():
    """Запуск Django-команд через командную строку."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            (
                "Не удалось импортировать Django. "
                "Убедитесь, что зависимости установлены."
            )
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
