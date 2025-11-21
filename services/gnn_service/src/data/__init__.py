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

# Will be added as we implement
# from src.data.timescale_connector import TimescaleConnector
# from src.data.feature_engineer import FeatureEngineer
# from src.data.graph_builder import GraphBuilder
# from src.data.dataset import HydraulicGraphDataset
# from src.data.loader import create_dataloader, create_train_val_loaders
from src.data.feature_config import FeatureConfig, DataLoaderConfig

__all__ = [
    # Configs
    "FeatureConfig",
    "DataLoaderConfig",
    # Components (will be added)
    # "TimescaleConnector",
    # "FeatureEngineer",
    # "GraphBuilder",
    # "HydraulicGraphDataset",
    # "create_dataloader",
    # "create_train_val_loaders",
]
