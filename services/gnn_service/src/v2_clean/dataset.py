"""PyTorch Dataset for node-centric hydraulic graphs.

Pipeline:
  Raw Data (UCI .txt files, 100 Hz)
    ↓ RawDataLoader
  Cycles (600 samples, 17 sensors)
    ↓ Resample 100 Hz → 10 Hz
  Preprocessed Cycles (600 samples @ 10 Hz)
    ↓ SemisyntheticFeatureEngineer
  Node Features (17 nodes × 48 features)
    ↓ NodeCentricTopology
  PyG Data Graph (x, edge_index, y)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch_geometric.data import Data, Dataset

from .data import Cycle, RawDataLoader, SENSOR_CONFIG, SENSOR_ORDER
from .features import SemisyntheticFeatureEngineer
from .topology import NodeCentricTopology

logger = logging.getLogger(__name__)


class NodeCentricGraphDataset(Dataset):
    """PyTorch Geometric dataset of node-centric hydraulic graphs.

    Each sample is a graph where:
      - Nodes (17): Sensors (PS1-PS6, TS1-TS4, FS1-FS2, VS1, SE, CE, CP, EPS1)
      - Node features (48): Semisynthetic features per sensor
      - Edges (22): Physical diagnostic connections
      - Labels (4): Multi-label classification (cooler, valve, pump, accumulator)

    Args:
        raw_data_dir: Path to UCI dataset (raw_real_dataset)
        split: "train", "val", or "test" (will be split by indices)
        normalize_features: Whether to normalize features to [0, 1]
        cache_features: Whether to cache computed features to disk (faster epoch 2+)
    """

    def __init__(
        self,
        raw_data_dir: str,
        split: str = "train",
        test_fraction: float = 0.15,
        val_fraction: float = 0.15,
        normalize_features: bool = True,
        cache_features: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.raw_data_dir = raw_data_dir
        self.split = split
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.normalize_features = normalize_features
        self.cache_features = cache_features

        # Initialize topology (fixed for all samples)
        self.topology = NodeCentricTopology()
        logger.info(f"Loaded topology: {self.topology}")

        # Load raw data
        logger.info(f"Loading raw data from {raw_data_dir}...")
        self.loader = RawDataLoader(raw_data_dir)
        self.cycles = self.loader.load_cycles()
        logger.info(f"Loaded {len(self.cycles)} total cycles")

        # Split into train/val/test
        self.indices = self._get_split_indices()
        logger.info(f"{split}: {len(self.indices)} cycles")

        # Compute feature normalization statistics (on train split)
        self.feature_stats = None
        if self.normalize_features:
            self.feature_stats = self._compute_feature_stats()

        super().__init__(None, transform, pre_transform)

    def _get_split_indices(self) -> list[int]:
        """Get indices for current split."""
        total = len(self.cycles)
        test_size = int(total * self.test_fraction)
        val_size = int(total * self.val_fraction)
        train_size = total - test_size - val_size

        # Simple split: first N for train, next M for val, rest for test
        if self.split == "train":
            return list(range(train_size))
        elif self.split == "val":
            return list(range(train_size, train_size + val_size))
        elif self.split == "test":
            return list(range(train_size + val_size, total))
        else:
            msg = f"Unknown split: {self.split}"
            raise ValueError(msg)

    def _compute_feature_stats(self) -> dict[str, tuple[float, float]]:
        """Compute mean and std of each feature (across train split for normalization)."""
        logger.info("Computing feature statistics...")

        # Get train indices
        train_indices = self._get_split_indices() if self.split == "train" else list(range(
            int(len(self.cycles) * (1 - self.test_fraction - self.val_fraction))
        ))

        all_features = []

        for idx in train_indices[:min(100, len(train_indices))]:  # Sample first 100 for speed
            cycle = self.cycles[idx]
            features = self._extract_graph_features(cycle)
            all_features.append(features.view(-1).numpy())  # Flatten to (17*48,)

        if not all_features:
            logger.warning("No features computed for statistics")
            return None

        all_features = np.array(all_features)  # (n_samples, 17*48)
        stats = {}
        for i in range(all_features.shape[1]):
            col = all_features[:, i]
            stats[f"feat_{i}"] = (np.mean(col), np.std(col) + 1e-6)  # Add eps to avoid div by zero

        logger.info(f"Computed statistics for {len(stats)} features")
        return stats

    def _extract_graph_features(self, cycle: Cycle) -> torch.Tensor:
        """Extract node features for a cycle.

        Args:
            cycle: Cycle object with data (600, 17)

        Returns:
            x: Tensor of shape (17, 48) with semisynthetic features
        """
        features_list = []

        for sensor_idx, sensor_name in enumerate(SENSOR_ORDER):
            # Get sensor readings for this cycle
            sensor_values = cycle.data[:, sensor_idx]  # (600,)

            # Get sensor metadata
            sensor_meta = SENSOR_CONFIG[sensor_name]

            # Extract semisynthetic features
            feature_eng = SemisyntheticFeatureEngineer(
                sensor_name=sensor_name,
                sensor_type=sensor_meta.sensor_type,
                physical_range=sensor_meta.physical_range,
                hz=10,  # After resampling
            )
            features = feature_eng.extract(sensor_values, metadata=cycle.metadata)
            features_list.append(features)

        # Stack all sensor features: (17, 48)
        x = torch.tensor(np.array(features_list), dtype=torch.float32)

        # Fill in relational features (correlations with adjacent sensors)
        x = self._fill_relational_features(x, cycle.data)

        # Normalize if requested
        if self.normalize_features and self.feature_stats:
            x = self._normalize_features(x)

        return x

    def _fill_relational_features(self, x: torch.Tensor, cycle_data: np.ndarray) -> torch.Tensor:
        """Fill in relational features (correlations with adjacent sensors).

        Args:
            x: Node features (17, 48)
            cycle_data: Raw cycle data (600, 17)

        Returns:
            x: Updated with correlations in positions 24-29
        """
        # For each node, compute correlations with adjacent nodes (from topology)
        for node_idx, node_name in enumerate(SENSOR_ORDER):
            adjacent_edges = self.topology.get_adjacent_edges(node_name)

            # Get correlations with up to 6 adjacent sensors
            for edge_idx, edge in enumerate(adjacent_edges[:6]):
                # Find adjacent sensor index
                adj_sensor_name = edge.target if edge.source == node_name else edge.source
                adj_sensor_idx = SENSOR_ORDER.index(adj_sensor_name)

                # Compute Pearson correlation
                corr, _ = scipy_stats.pearsonr(
                    cycle_data[:, node_idx], cycle_data[:, adj_sensor_idx]
                )
                corr = np.nan_to_num(corr, nan=0.0)  # Handle NaN (constant signals)

                # Store in relational feature position
                x[node_idx, 24 + edge_idx] = float(corr)

        return x

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features using pre-computed statistics."""
        if self.feature_stats is None:
            return x

        x_norm = x.clone()
        for feat_idx in range(x.shape[1]):
            key = f"feat_{feat_idx}"
            if key in self.feature_stats:
                mean, std = self.feature_stats[key]
                x_norm[:, feat_idx] = (x[:, feat_idx] - mean) / std

        return x_norm

    def len(self) -> int:
        """Number of samples in this split."""
        return len(self.indices)

    def get(self, idx: int) -> Data:
        """Get a single graph sample.

        Args:
            idx: Index in current split

        Returns:
            data: PyG Data object with x, edge_index, y
        """
        # Get actual cycle index (accounting for split)
        cycle_idx = self.indices[idx]
        cycle = self.cycles[cycle_idx]

        # Extract node features
        x = self._extract_graph_features(cycle)

        # Get edge index
        source_indices, target_indices = self.topology.get_edge_index()
        edge_index = torch.tensor(
            [source_indices, target_indices], dtype=torch.long
        )

        # Get labels (multi-label)
        label = cycle.label
        y = torch.tensor(
            [label.cooler, label.valve, label.pump_leak, label.accumulator],
            dtype=torch.long,
        )

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            cycle_id=torch.tensor(cycle.cycle_id),
            metadata=cycle.metadata,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data
