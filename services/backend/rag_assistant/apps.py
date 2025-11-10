"""Конфигурация приложения Django для RAG Assistant."""

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class RagAssistantConfig(AppConfig):
    """Конфигурация приложения RAG Assistant."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_assistant"
    verbose_name = "RAG Ассистент"

    def ready(self):
        """Выполняет инициализацию приложения при запуске Django.

        Проверяет зависимости и настраивает сигналы.
        """
        try:
            # Импорт сигналов
            pass

            # Проверка зависимостей
            self._check_dependencies()

            # Используем стандартное логирование без structlog
            print("✅ RAG Ассистент приложение инициализировано")

        except Exception as e:
            print(f"❌ Ошибка инициализации RAG Ассистента: {e}")

    def _check_dependencies(self):
        """Проверяет наличие необходимых зависимостей.

        Выводит предупреждения об отсутствующих пакетах.
        """
        required_packages = ["sentence_transformers", "sklearn", "numpy", "pandas"]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            # Используем print вместо logger для избежания конфликтов
            print(f"⚠️ Отсутствуют пакеты: {', '.join(missing_packages)}")
            print(
                "ℹ️ Установите их: pip install sentence-transformers scikit-learn numpy pandas"
            )
