"""
Unified and dynamic configuration for Universal GNN Service.

Centralized config management:
- ModelConfig - GNN model hyperparameters
- TrainingConfig - Training settings
- DBConfig - Database connection
- APIConfig - FastAPI settings (port 8000)
- InferenceConfig - Inference engine settings (NEW)
- ObservabilityConfig - Logging configuration
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ModelConfig:
    """GNN Model hyperparameters."""
    hidden_dim: int = 128
    num_heads: int = 8
    num_gat_layers: int = 3
    lstm_hidden_dim: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    data_path: str = "data/bim_comprehensive.csv"
    metadata_path: str = "data/equipment_metadata.json"
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 100
    learning_rate: float = 1e-3
    window_minutes: int = 60
    timestep_minutes: int = 5
    sequence_length: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class DBConfig:
    """TimescaleDB connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "hydraulic_db"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    timeout: float = 5.0


@dataclass
class APIConfig:
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000  # Production API port
    reload: bool = False
    workers: int = 2
    model_path: str = "models/v2.0.0.ckpt"
    metadata_path: str = "data/equipment_metadata.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferenceConfig:
    """Inference engine configuration.
    
    All values loaded from environment with GNN_ prefix.
    No hardcoded values for production deployment.
    
    Example .env:
        GNN_INFERENCE_TIMEOUT_S=30.0
        GNN_BATCH_SIZE=32
        GNN_QUEUE_PUT_TIMEOUT_S=5.0
    """
    # Timeouts (seconds)
    inference_timeout_s: float = float(os.getenv("GNN_INFERENCE_TIMEOUT_S", "30.0"))
    queue_put_timeout_s: float = float(os.getenv("GNN_QUEUE_PUT_TIMEOUT_S", "5.0"))
    queue_collect_timeout_ms: float = float(os.getenv("GNN_QUEUE_COLLECT_TIMEOUT_MS", "50.0"))
    
    # Batch processing
    batch_size: int = int(os.getenv("GNN_BATCH_SIZE", "32"))
    max_queue_size: int = int(os.getenv("GNN_MAX_QUEUE_SIZE", "100"))
    enable_dynamic_batching: bool = os.getenv("GNN_ENABLE_DYNAMIC_BATCHING", "false").lower() == "true"
    
    # Model settings
    model_path: str | None = os.getenv("GNN_MODEL_PATH")
    device: str = os.getenv("GNN_DEVICE", "auto")
    fallback_to_cpu: bool = os.getenv("GNN_FALLBACK_TO_CPU", "true").lower() == "true"
    enable_compile: bool = os.getenv("GNN_ENABLE_COMPILE", "true").lower() == "true"
    pin_memory: bool = os.getenv("GNN_PIN_MEMORY", "true").lower() == "true"
    
    # Features
    use_dynamic_features: bool = os.getenv("GNN_USE_DYNAMIC_FEATURES", "true").lower() == "true"
    use_dynamic_builder: bool = os.getenv("GNN_USE_DYNAMIC_BUILDER", "true").lower() == "true"
    
    # Topology & Caching
    topology_templates_path: Path | None = Path(p) if (p := os.getenv("GNN_TOPOLOGY_TEMPLATES_PATH")) else None
    topology_cache_size: int = int(os.getenv("GNN_TOPOLOGY_CACHE_SIZE", "100"))
    topology_cache_ttl_s: float = float(os.getenv("GNN_TOPOLOGY_CACHE_TTL_S", "300.0"))
    
    # Validation
    validate_tensors: bool = os.getenv("GNN_VALIDATE_TENSORS", "true").lower() == "true"


@dataclass
class ObservabilityConfig:
    """Logging and monitoring configuration."""
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: str = "logs"


# Global singleton instances
model_config = ModelConfig()
training_config = TrainingConfig()
db_config = DBConfig()
api_config = APIConfig()
inference_config = InferenceConfig()
observability_config = ObservabilityConfig()
