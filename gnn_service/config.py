"""Configuration for GNN Service."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GNNConfig:
    """GNN Training Configuration."""
    
    # Model architecture
    hidden_dim: int = 128
    num_gat_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    temporal_window: int = 10  # timesteps
    
    # SSL Pretraining
    ssl_epochs: int = 30
    ssl_mask_ratio: float = 0.15
    ssl_contrastive_temp: float = 0.07
    ssl_lr: float = 1e-3
    
    # Supervised Fine-tuning
    finetune_epochs: int = 70
    finetune_lr: float = 5e-4
    batch_size: int = 32
    
    # Data
    num_node_features: int = 50  # sensor features per component
    num_classes: int = 2  # normal vs anomaly
    
    # Hardware
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    num_workers: int = 4
    
    # Database
    postgres_host: str = os.environ.get("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.environ.get("POSTGRES_PORT", "5432"))
    postgres_db: str = os.environ.get("POSTGRES_DB", "hydraulic_diagnostics")
    postgres_user: str = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password: str = os.environ.get("POSTGRES_PASSWORD", "")
    
    # Paths
    model_save_dir: str = "./models"
    log_dir: str = "./logs"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8003
    
    @property
    def postgres_uri(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


config = GNNConfig()
