"""Django app configuration for diagnostics application."""

from django.apps import AppConfig


class DiagnosticsConfig(AppConfig):
    """Configuration for diagnostics application."""
    
    default_auto_field = "django.db.models.BigAutoField"
    name = "diagnostics"  # FIXED: было apps.diagnostics
    verbose_name = "Hydraulic Diagnostics"

    def ready(self) -> None:
        """Initialize application when Django starts.
        
        This is called when Django starts. Use it to:
        - Register signals
        - Import models
        - Initialize caches
        - Setup connections
        """
        # Import signals (if any)
        try:
            import diagnostics.signals  # noqa: F401
        except ImportError:
            pass
        
        # Register admin customizations
        try:
            from diagnostics import admin_quarantine  # noqa: F401
        except ImportError:
            pass
