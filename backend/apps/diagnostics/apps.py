from django.apps import AppConfig


class DiagnosticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.diagnostics"
    verbose_name = "Диагностические Системы"

    def ready(self):
        """Инициализация при запуске приложения"""
        try:
            # Импорт AI системы для инициализации
            from .ai_engine import ai_engine
            from .rag_system import rag_system

            print("✅ AI Engine и RAG система инициализированы")
        except Exception as e:
            print(f"❌ Ошибка инициализации AI/RAG: {e}")
