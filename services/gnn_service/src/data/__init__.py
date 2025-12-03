"""Data module for hydraulic graph dataset.

Components:
- TimescaleConnector - async database access
- FeatureEngineer - feature extraction pipeline
- GraphBuilder - sensor data → PyG graphs
- HydraulicGraphDataset - PyTorch Dataset
- DataLoader factory - efficient batching

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from src.data.dataset import HydraulicGraphDataset

# Configuration
from src.data.feature_config import DataLoaderConfig, FeatureConfig
from src.data.feature_engineer import FeatureEngineer
from src.data.graph_builder import GraphBuilder
from src.data.loader import (
    create_dataloader,
    create_train_val_loaders,
    create_train_val_test_loaders,
    hydraulic_collate_fn,
)

# Components
from src.data.timescale_connector import TimescaleConnector

__all__ = [
    # Configs
    "FeatureConfig",
    "DataLoaderConfig",
    # Components
    "TimescaleConnector",
    "FeatureEngineer",
    "GraphBuilder",
    "HydraulicGraphDataset",
    # DataLoader utilities
    "hydraulic_collate_fn",
    "create_dataloader",
    "create_train_val_loaders",
    "create_train_val_test_loaders",
]
