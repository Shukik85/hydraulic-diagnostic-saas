"""
Unified and dynamic configuration for Universal GNN Service.

Centralized config management:
- ModelConfig - GNN model hyperparameters
- TrainingConfig - Training settings
- DBConfig - Database connection
- APIConfig - FastAPI settings (port 8000)
- ObservabilityConfig - Logging configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
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
observability_config = ObservabilityConfig()
