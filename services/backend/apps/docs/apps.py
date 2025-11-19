"""App configuration for documentation system."""

from django.apps import AppConfig


class DocsConfig(AppConfig):
    """Configuration for docs app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.docs"
    verbose_name = "Documentation & Guides"

    def ready(self) -> None:
        """Import signals when app is ready."""
        # Future: import signals if needed
        pass
