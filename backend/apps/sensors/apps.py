"""
Sensors Django App Configuration.

Handles industrial sensor data ingestion via multiple protocols:
- Modbus TCP/RTU
- OPC UA 
- Raw HTTP/JSON
- MQTT (future)

TimescaleDB optimized for high-frequency sensor data storage.
"""

from django.apps import AppConfig


class SensorsConfig(AppConfig):
    """Sensors app configuration for industrial data ingestion."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.sensors'
    verbose_name = 'Industrial Sensors'
    
    def ready(self) -> None:
        """Initialize sensors app - register signal handlers and protocols."""
        try:
            # Import signal handlers
            from . import signals  # noqa: F401
            
            # Initialize protocol handlers
            from .protocols import initialize_protocol_registry
            initialize_protocol_registry()
            
        except ImportError:
            # Graceful handling during migrations
            pass
