"""
Production-grade configuration management using Pydantic Settings v2.

Ключевые улучшения:
- Type-safe configuration с runtime валидацией
- Автоматический парсинг environment variables
- .env file support
- Nested configuration структура
- Validators для custom logic
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==================== Model Configuration ====================
class ModelConfig(BaseSettings):
    """GNN model architecture configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
    )

    # Architecture params
    hidden_dim: int = Field(
        default=128, ge=32, le=512, description="Hidden dimension size"
    )
    num_heads: int = Field(
        default=8, ge=1, le=16, description="Number of attention heads"
    )
    num_gat_layers: int = Field(
        default=3, ge=1, le=6, description="Number of GAT layers"
    )
    lstm_hidden_dim: int = Field(
        default=256, ge=64, le=1024, description="LSTM hidden size"
    )
    lstm_layers: int = Field(default=2, ge=1, le=4, description="Number of LSTM layers")
    dropout: float = Field(default=0.3, ge=0.0, le=0.8, description="Dropout rate")

    # Device configuration
    device: Literal["cuda", "cpu", "mps"] = Field(default="cuda")
    compile_model: bool = Field(
        default=True, description="Use torch.compile for optimization"
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="max-autotune"
    )

    # Model paths
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    model_path: Path = Field(default=Path("models/best_model.tar"))

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device availability."""
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if v == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return v

    @model_validator(mode="after")
    def validate_paths(self) -> ModelConfig:
        """Ensure required directories exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.model_path.parent != Path():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        return self


# ==================== Training Configuration ====================
class TrainingConfig(BaseSettings):
    """Training pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",
        env_file=".env",
        case_sensitive=False,
    )

    # Data paths
    data_path: Path = Field(default=Path("data/bim_comprehensive.csv"))
    metadata_path: Path = Field(default=Path("data/equipment_metadata.json"))

    # Training hyperparameters
    batch_size: int = Field(default=16, ge=1, le=128)
    num_workers: int = Field(default=4, ge=0, le=16)
    max_epochs: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-1)
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1e-2)

    # Early stopping
    early_stopping_patience: int = Field(default=10, ge=1)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0.0)

    # Data processing
    window_minutes: int = Field(default=60, ge=5, le=1440)
    timestep_minutes: int = Field(default=5, ge=1, le=60)
    sequence_length: int = Field(default=5, ge=1, le=20)

    # Data splits
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    # Mixed precision
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=16)

    # Distributed training
    distributed: bool = Field(default=False)
    world_size: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def validate_splits(self) -> TrainingConfig:
        """Validate train/val/test splits sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total}")
        return self


# ==================== Database Configuration ====================
class DatabaseConfig(BaseSettings):
    """TimescaleDB connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
    )

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="hydraulic_db")
    user: str = Field(default="postgres")
    password: str = Field(default="", description="Database password")

    # Connection pool settings
    pool_min_size: int = Field(default=2, ge=1, le=10)
    pool_max_size: int = Field(default=10, ge=1, le=100)
    pool_timeout: float = Field(default=5.0, ge=1.0, le=60.0)
    pool_recycle: int = Field(
        default=3600, ge=300, description="Recycle connections after N seconds"
    )

    # Query settings
    query_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    statement_timeout: int = Field(default=30000, description="Statement timeout in ms")

    # Health check
    health_check_interval: int = Field(default=30, ge=10, le=300)

    @property
    def dsn(self) -> str:
        """Construct PostgreSQL DSN."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        """Construct async PostgreSQL DSN."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# ==================== API Configuration ====================
class APIConfig(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
    )

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002, ge=1024, le=65535)
    workers: int = Field(default=2, ge=1, le=16)
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # CORS
    cors_origins: list[str] = Field(default=["*"])
    cors_credentials: bool = Field(default=True)
    cors_methods: list[str] = Field(default=["*"])
    cors_headers: list[str] = Field(default=["*"])

    # Request limits
    max_request_size: int = Field(default=10 * 1024 * 1024, description="10MB")
    request_timeout: float = Field(default=30.0, ge=1.0, le=300.0)

    # Inference settings
    inference_batch_size: int = Field(default=16, ge=1, le=100)
    inference_batch_timeout_ms: float = Field(default=50.0, ge=10.0, le=1000.0)
    inference_queue_size: int = Field(default=100, ge=10, le=1000)

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, description="Requests per minute")

    # Cache settings
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=300, ge=0, description="Cache TTL in seconds")


# ==================== Observability Configuration ====================
class ObservabilityConfig(BaseSettings):
    """Logging, metrics, and tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OBS_",
        env_file=".env",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "text"] = Field(default="json")
    log_dir: Path = Field(default=Path("logs"))
    log_rotation: str = Field(default="100 MB", description="Log file rotation size")
    log_retention: str = Field(default="30 days", description="Log retention period")

    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    metrics_path: str = Field(default="/metrics")

    # Tracing
    tracing_enabled: bool = Field(default=True)
    tracing_endpoint: str = Field(default="http://localhost:4317")
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Service info
    service_name: str = Field(default="gnn-service")
    service_version: str = Field(default="2.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="production"
    )

    @model_validator(mode="after")
    def create_log_dir(self) -> ObservabilityConfig:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


# ==================== Root Configuration ====================
class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Global settings
    debug: bool = Field(default=False)
    testing: bool = Field(default=False)

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


# ==================== Singleton Pattern ====================
@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses LRU cache to ensure single instance across application.
    Settings are loaded once and reused.
    """
    return Settings()


# ==================== Export ====================
__all__ = [
    "Settings",
    "ModelConfig",
    "TrainingConfig",
    "DatabaseConfig",
    "APIConfig",
    "ObservabilityConfig",
    "get_settings",
]


# ==================== Usage Example ====================
if __name__ == "__main__":
    settings = get_settings()

    print("=== GNN Service Configuration ===")
    print(f"Environment: {settings.observability.environment}")
    print(f"Device: {settings.model.device}")
    print(f"API Port: {settings.api.port}")
    print(f"Database: {settings.database.dsn}")
    print(f"Log Level: {settings.observability.log_level}")

    # Validate configuration
    print("\n✅ All configurations validated successfully!")
