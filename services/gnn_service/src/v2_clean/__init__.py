"""V2 Clean Architecture: Node-Centric Graphs + Semisynthetic Features.

This module contains production-ready implementation with:
- Raw data loading from UCI dataset
- Node-centric topology (sensors are nodes, hydraulic lines are edges)
- Semisynthetic features (raw values + engineered)
- Direct training pipeline without intermediate abstractions

Structure:
  data/           - Raw data loading and preprocessing
  topology/       - Node-centric graph topology definitions
  features/       - Semisynthetic feature engineering
  dataset/        - PyTorch dataset implementation
  training/       - Training loop (coming soon)

Quick Start:
  >>> from src.v2_clean.data import RawDataLoader
  >>> from src.v2_clean.dataset import NodeCentricGraphDataset
  >>>
  >>> loader = RawDataLoader(data_dir="data/raw_real_dataset")
  >>> cycles = loader.load_cycles()
  >>>
  >>> dataset = NodeCentricGraphDataset(cycles)
  >>> len(dataset)  # 2600 cycles
  >>> graph = dataset[0]
  >>> graph.x.shape  # [17 sensors, 48 features]
  >>> graph.edge_index.shape  # [2, num_connections]
"""
