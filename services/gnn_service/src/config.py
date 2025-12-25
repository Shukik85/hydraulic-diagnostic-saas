"""Global configuration from environment variables.

Loads settings from .env file using pydantic-settings.
Provides global `settings` instance for application-wide config.

Python 3.14 features:
    - Native type annotations with | operator
    - Improved Path handling

Examples:
    >>> from src.config import settings
    >>> if settings.use_timescaledb_mock:
    ...     adapter = TimescaleDBMockAdapter()
    ... else:
    ...     adapter = TimescaleDBRealAdapter()
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    """Global settings from .env file.

    Automatically loads from .env file in service root directory.
    Case-insensitive environment variable names.

    Attributes:
        TimescaleDB Mock/Real:
            use_timescaledb_mock: Use mock adapter for testing
            timescaledb_*: Real database connection parameters
            mock_timescaledb_*: Mock adapter configuration

        Data Sources:
            csv_data_dir: Directory with CSV sensor data
            api_base_url: REST API base URL

        Sensor Defaults:
            default_*_sampling_rate: Default Hz for sensor types

        Value Substitution:
            enable_value_substitution: Enable physics estimation
            fluid_*: Hydraulic fluid properties
            material_roughness_*: Pipe/hose roughness by material

        Logging:
            log_level: Logging verbosity
            log_format: Log format (json/text)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # TIMESCALEDB CONFIGURATION
    # =========================================================================

    # Mock/Real toggle
    use_timescaledb_mock: bool = True

    # Real TimescaleDB connection
    timescaledb_host: str = "localhost"
    timescaledb_port: int = 5432
    timescaledb_database: str = "hydraulic_diagnostics"
    timescaledb_user: str = "postgres"
    timescaledb_password: str = ""
    timescaledb_pool_size: int = 10
    timescaledb_max_overflow: int = 20

    # Mock TimescaleDB
    mock_timescaledb_seed: int = 42
    mock_timescaledb_noise_level: float = 0.05

    # =========================================================================
    # DATA SOURCES
    # =========================================================================

    # CSV
    csv_data_dir: Path = Path("./data/sensor_readings")
    csv_timestamp_format: str = "%Y-%m-%dT%H:%M:%S%z"

    # REST API
    api_base_url: str = "http://localhost:8080/api/v1"
    api_timeout_seconds: int = 5
    api_max_retries: int = 3

    # =========================================================================
    # SENSOR DEFAULTS
    # =========================================================================

    default_pressure_sampling_rate: float = 10.0
    default_flow_sampling_rate: float = 5.0
    default_temperature_sampling_rate: float = 1.0
    default_rpm_sampling_rate: float = 1.0

    # =========================================================================
    # VALUE SUBSTITUTION ENGINE
    # =========================================================================

    # Enable/disable
    enable_value_substitution: bool = True

    # Fluid properties (ISO VG 46 hydraulic oil at 40°C)
    fluid_density_kg_m3: float = 870.0
    fluid_viscosity_pas: float = 0.046

    # Material roughness (mm)
    material_roughness_steel: float = 0.045
    material_roughness_rubber: float = 0.15
    material_roughness_composite: float = 0.1

    # =========================================================================
    # LOGGING
    # =========================================================================

    log_level: str = "INFO"
    log_format: str = "json"


# Global settings instance
settings = Settings()
