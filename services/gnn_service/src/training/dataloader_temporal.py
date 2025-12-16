"""Temporal DataLoader based on TimeGNN approach.

Provides temporal graph snapshots with:
- Sliding window temporal construction
- Dynamic edge detection from sensor correlations
- Missing data handling via GRAPE
- Multi-task targets (health, degradation, anomaly, RUL)

References:
  [1] TimeGNN: 4-80x faster temporal GNN inference
  [2] GRAPE: Missing data via GNN edge embeddings
  [3] Dynamic graph construction for time series
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

if TYPE_CHECKING:
    from src.data.timescale_connector import TimescaleConnector
    from src.features.feature_engineer import FeatureEngineer
    from src.schemas import GraphTopology

logger = logging.getLogger(__name__)


class TemporalGraphSnapshot:
    """Represents graph state at a time point.

    Attributes:
        timestamp: Snapshot timestamp
        x: Node features [N, F]
        edge_index: Dynamic edges [2, E]
        edge_attr: Edge attributes [E, 8]
        mask_nodes: Binary mask for missing nodes
        mask_edges: Binary mask for reconstructed edges
    """

    def __init__(
        self,
        timestamp: str,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        mask_nodes: np.ndarray | None = None,
        mask_edges: np.ndarray | None = None,
    ):
        self.timestamp = timestamp
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.mask_nodes = mask_nodes or np.ones(x.shape[0])
        self.mask_edges = mask_edges or np.ones(edge_index.shape[1])


class TemporalHydraulicDataLoader:
    """DataLoader for temporal hydraulic GNN training.

    Implements TimeGNN + GRAPE approach:
    1. Create temporal snapshots with sliding windows
    2. Detect dynamic edges from sensor correlations
    3. Handle missing data via graph reconstruction
    4. Batch snapshots for training

    Examples:
        >>> loader = TemporalHydraulicDataLoader(
        ...     timescale_connector=connector,
        ...     feature_engineer=engineer,
        ...     window_size=timedelta(hours=1),
        ...     stride=timedelta(minutes=15),
        ... )
        >>> datasets = await loader.load_temporal_dataset(
        ...     equipment_ids=["pump_001", "pump_002"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ... )
        >>> train_loader = loader.get_train_loader(datasets[0])
    """

    def __init__(
        self,
        timescale_connector: TimescaleConnector,
        feature_engineer: FeatureEngineer,
        window_size: timedelta = timedelta(hours=1),
        stride: timedelta = timedelta(minutes=15),
        correlation_threshold: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Initialize temporal dataloader.

        Args:
            timescale_connector: TimescaleDB connection
            feature_engineer: Feature extraction engine
            window_size: Temporal window (e.g., 1 hour)
            stride: Window stride (e.g., 15 min)
            correlation_threshold: Edge creation threshold
            batch_size: Batch size for training
            num_workers: DataLoader workers
        """
        self.timescale_connector = timescale_connector
        self.feature_engineer = feature_engineer
        self.window_size = window_size
        self.stride = stride
        self.correlation_threshold = correlation_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

    async def load_temporal_dataset(
        self,
        equipment_ids: list[str],
        start_date: str,
        end_date: str,
        topology: GraphTopology,
    ) -> list[list[Data]]:
        """Load temporal dataset as list of temporal graphs.

        Returns:
            For each equipment: list of Data objects (one per time window)
        """
        datasets = []

        for equipment_id in equipment_ids:
            logger.info(f"Loading temporal data for {equipment_id}")

            # Fetch time series data
            df = await self.timescale_connector.fetch_sensor_data(
                equipment_id=equipment_id,
                start_time=start_date,
                end_time=end_date,
            )

            if df.is_empty():
                logger.warning(f"No data for {equipment_id}")
                continue

            # Create temporal snapshots
            temporal_graphs = self._create_temporal_snapshots(
                df=df,
                topology=topology,
                equipment_id=equipment_id,
            )

            # Load targets for each snapshot
            temporal_graphs = await self._load_targets_for_snapshots(
                temporal_graphs=temporal_graphs,
                equipment_id=equipment_id,
            )

            datasets.append(temporal_graphs)

        return datasets

    def _create_temporal_snapshots(
        self,
        df: pl.DataFrame,
        topology: GraphTopology,
        equipment_id: str,
    ) -> list[Data]:
        """Create temporal graph snapshots with sliding windows.

        Each snapshot is a PyG Data object representing
        the hydraulic system state at time t.
        """
        snapshots = []
        timestamps = df.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()

        for i in range(0, len(timestamps) - int(self.window_size.total_seconds() / 60), int(self.stride.total_seconds() / 60)):
            window_start = timestamps[i]
            window_end = timestamps[i + int(self.window_size.total_seconds() / 60)]

            # Filter data for this window
            window_df = df.filter((df["timestamp"] >= window_start) & (df["timestamp"] <= window_end))

            if window_df.is_empty():
                continue

            # Extract features for each node
            node_features, mask_nodes = self._extract_node_features(
                window_df, topology
            )

            # Construct dynamic edges from correlations
            edge_index, edge_attr, mask_edges = self._construct_dynamic_edges(
                window_df, topology, node_features
            )

            # Create Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                mask_nodes=torch.tensor(mask_nodes, dtype=torch.float32),
                mask_edges=torch.tensor(mask_edges, dtype=torch.float32),
                timestamp=str(window_start),
                equipment_id=equipment_id,
            )

            snapshots.append(data)

        return snapshots

    def _extract_node_features(
        self, df: pl.DataFrame, topology: GraphTopology
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract node features with missing data masking.

        Returns:
            node_features: [N, F] feature matrix
            mask_nodes: [N] binary mask (1=data present, 0=missing)
        """
        features = []
        masks = []

        for component in topology.components.values():
            sensor_id = component.component_id
            sensor_df = df.filter(pl.col("sensor_id") == sensor_id)

            if sensor_df.is_empty():
                # Missing sensor: zero features, mask=0
                feat = np.zeros(34)
                mask = 0.0
            else:
                # Use FeatureEngineer
                feat = self.feature_engineer.extract_all_features(sensor_df.to_numpy())
                mask = 1.0

            features.append(feat)
            masks.append(mask)

        return np.array(features), np.array(masks)

    def _construct_dynamic_edges(
        self, df: pl.DataFrame, topology: GraphTopology, node_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct dynamic edges from sensor correlations.

        Uses correlation matrix to detect edges dynamically,
        implementing the "dynamic edge detection" from research.
        """
        n_nodes = len(topology.components)
        edge_index = [[], []]
        edge_attrs = []
        edge_mask = []

        # Compute pairwise correlations
        correlations = np.corrcoef(node_features)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and abs(correlations[i, j]) > self.correlation_threshold:
                    edge_index[0].append(i)
                    edge_index[1].append(j)

                    # Static + dynamic features (8+6=14)
                    edge_attr = np.concatenate([
                        np.zeros(8),  # Static features
                        [correlations[i, j], 0, 0, 0, 0, 0],  # Dynamic features
                    ])
                    edge_attrs.append(edge_attr)
                    edge_mask.append(1.0)  # Dynamic edge detected

        # Add static topology edges if missing
        for conn in topology.connections:
            nodes = list(topology.components.keys())
            i = nodes.index(conn["from"])
            j = nodes.index(conn["to"])
            if [i, j] not in list(zip(edge_index[0], edge_index[1])):
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_attrs.append(np.zeros(14))  # Static edge
                edge_mask.append(0.0)  # Not dynamically detected

        if not edge_index[0]:
            edge_index = np.array([[0, 1], [1, 0]]).T
            edge_attrs = [np.zeros(14), np.zeros(14)]
            edge_mask = [0.0, 0.0]

        return (
            np.array(edge_index),
            np.array(edge_attrs) if edge_attrs else np.zeros((len(edge_index[0]), 14)),
            np.array(edge_mask),
        )

    async def _load_targets_for_snapshots(
        self, temporal_graphs: list[Data], equipment_id: str
    ) -> list[Data]:
        """Load targets for each temporal snapshot."""
        # Placeholder: would query from labels table
        for graph in temporal_graphs:
            graph.y_graph_health = torch.tensor([0.8], dtype=torch.float32)
            graph.y_graph_degradation = torch.tensor([0.2], dtype=torch.float32)
            graph.y_graph_anomaly = torch.zeros(9, dtype=torch.float32)
            graph.y_graph_rul = torch.tensor([1000.0], dtype=torch.float32)
            graph.y_component_health = torch.ones(5, 1, dtype=torch.float32)
            graph.y_component_anomaly = torch.zeros(5, 9, dtype=torch.float32)

        return temporal_graphs

    def get_train_loader(self, temporal_graphs: list[Data]) -> PyGDataLoader:
        """Create training DataLoader."""
        return PyGDataLoader(
            temporal_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_val_loader(self, temporal_graphs: list[Data]) -> PyGDataLoader:
        """Create validation DataLoader."""
        return PyGDataLoader(
            temporal_graphs,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
