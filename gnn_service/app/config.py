"""Configuration management для GNN Service."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """GNN Service settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # API Authentication
    gnn_internal_api_key: str = "changeme-gnn-secret-key"
    
    # Model Configuration
    model_path: str = "/models/gnn_classifier_best.ckpt"
    device: str = "cuda"  # cuda, cpu, mps
    
    # Inference Settings
    batch_size: int = 32
    num_workers: int = 4
    
    # T-GAT Model Hyperparameters
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    
    # Attention Explainability
    attention_threshold: float = 0.3  # Минимальный attention score для explanation
    
    # Logging
    log_level: str = "INFO"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
