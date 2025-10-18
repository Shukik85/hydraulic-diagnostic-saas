import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class RagAssistantConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.rag_assistant"
    verbose_name = "RAG Ассистент"

    def ready(self):
        """Инициализация приложения"""
        try:
            # Импорт сигналов
            pass

            # Проверка зависимостей
            self._check_dependencies()

            logger.info("RAG Ассистент приложение инициализировано")

        except Exception as e:
            logger.error(f"Ошибка инициализации RAG Ассистента: {e}")

    def _check_dependencies(self):
        """Проверка необходимых зависимостей"""
        required_packages = ["sentence_transformers", "sklearn", "numpy", "pandas"]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
            logger.warning(
                "Установите их: pip install sentence-transformers scikit-learn numpy pandas"
            )
