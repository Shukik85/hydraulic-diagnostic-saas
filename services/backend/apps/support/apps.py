"""Support app configuration."""

from django.apps import AppConfig


class SupportConfig(AppConfig):
    """Configuration for Support Management app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.support"
    verbose_name = "Support Management"

    def ready(self) -> None:
        """Import signals when app is ready."""
        # Import signals here if needed
        pass
