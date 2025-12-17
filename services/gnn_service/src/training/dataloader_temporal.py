"""Temporal dataloader for hydraulic diagnostics.

Provides:
- Temporal snapshot creation
- Dynamic edge construction
- Missing data handling
- Target loading from TimescaleDB
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from src.data.timescale_connector import TimescaleConnector
    from src.topology.graph_topology import GraphTopology

logger = logging.getLogger(__name__)


class TemporalHydraulicDataLoader:
    """DataLoader for temporal hydraulic diagnostics.

    Creates temporal snapshots with dynamic graph construction.

    Examples:
        >>> loader = TemporalHydraulicDataLoader(
        ...     timescale_connector=connector,
        ...     window_size=3600,
        ...     stride=900,
        ...     sequence_length=12,
        ... )
        >>> graphs = await loader.load_temporal_sequence(
        ...     equipment_id="excavator_001",
        ...     start_time="2024-01-01",
        ...     end_time="2024-01-02",
        ...     topology=graph_topology,
        ... )
    """

    def __init__(
        self,
        timescale_connector: TimescaleConnector,
        window_size: int = 3600,
        stride: int = 900,
        sequence_length: int = 12,
        correlation_threshold: float = 0.5,
        k_neighbors: int = 5,
        max_missing_ratio: float = 0.5,
    ):
        self.timescale = timescale_connector
        self.window_size = window_size
        self.stride = stride
        self.sequence_length = sequence_length
        self.correlation_threshold = correlation_threshold
        self.k_neighbors = k_neighbors
        self.max_missing_ratio = max_missing_ratio

    async def load_temporal_sequence(
        self,
        equipment_id: str,
        start_time: str,
        end_time: str,
        topology: GraphTopology,
    ) -> list[Data]:
        """Load temporal sequence of graphs.

        Args:
            equipment_id: Equipment identifier
            start_time: Start timestamp
            end_time: End timestamp
            topology: Graph topology

        Returns:
            List of temporal graph snapshots
        """
        # Load sensor data
        df = await self.timescale.fetch_sensor_data(equipment_id, start_time, end_time)

        if df.is_empty():
            logger.warning(f"No data found for {equipment_id} in [{start_time}, {end_time}]")
            return []

        # Create temporal windows
        temporal_graphs = self._create_temporal_snapshots(df, topology)

        # Load targets
        temporal_graphs = await self._load_targets_for_snapshots(temporal_graphs)

        return temporal_graphs

    def _create_temporal_snapshots(
        self, df: pl.DataFrame, topology: GraphTopology
    ) -> list[Data]:
        """Create temporal snapshots from sensor data.

        Args:
            df: Sensor data
            topology: Graph topology

        Returns:
            List of graph snapshots
        """
        # Get time range
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()

        # Generate window starts
        window_starts = []
        current_time = min_time
        while current_time + self.window_size <= max_time:
            window_starts.append(current_time)
            current_time += self.stride

        # Create snapshots
        snapshots = []
        for window_start in window_starts:
            window_end = window_start + self.window_size

            # Filter data for window
            window_df = df.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < window_end)
            )

            if window_df.is_empty():
                continue

            # Extract features
            node_features, mask_nodes = self._extract_node_features(window_df, topology)

            # Check missing ratio
            missing_ratio = 1.0 - mask_nodes.sum() / len(mask_nodes)
            if missing_ratio > self.max_missing_ratio:
                logger.warning(
                    f"Skipping window (missing ratio: {missing_ratio:.2%} > {self.max_missing_ratio:.2%})"
                )
                continue

            # Construct edges
            edge_index, edge_attr, edge_mask = self._construct_dynamic_edges(
                topology, node_features
            )

            # Create graph
            snapshot = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                mask_nodes=torch.tensor(mask_nodes, dtype=torch.bool),
                edge_mask=torch.tensor(edge_mask, dtype=torch.bool),
                timestamp=window_start,
            )

            snapshots.append(snapshot)

        return snapshots

    def _extract_node_features(
        self, df: pl.DataFrame, topology: GraphTopology
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract node features from sensor data.

        Args:
            df: Sensor data for time window
            topology: Graph topology

        Returns:
            node_features: [N, F] array
            mask_nodes: [N] bool array (True=observed, False=missing)
        """
        nodes = list(topology.components.keys())
        n_nodes = len(nodes)
        n_features = 34  # Fixed feature dimension

        node_features = np.zeros((n_nodes, n_features))
        mask_nodes = np.zeros(n_nodes, dtype=bool)

        for idx, node_id in enumerate(nodes):
            # Get sensor data for this component
            node_data = df.filter(pl.col("component_id") == node_id)

            if node_data.is_empty():
                # Missing node
                mask_nodes[idx] = False
                continue

            # Extract features (simplified - real implementation uses FeatureEngineer)
            # Features: mean, std, min, max, etc.
            values = node_data["value"].to_numpy()
            if len(values) > 0:
                node_features[idx, :4] = [
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                ]
                mask_nodes[idx] = True

        return node_features, mask_nodes

    def _construct_dynamic_edges(
        self, topology: GraphTopology, node_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct dynamic edges from sensor correlations.

        Args:
            topology: Graph topology (static edges)
            node_features: Node features [N, F]

        Returns:
            edge_index: [2, E]
            edge_attr: [E, 14]
            edge_mask: [E] bool (True=static, False=dynamic)
        """
        n_nodes = len(topology.components)

        # Start with static edges from topology
        edge_index = [[], []]
        edge_attr = []
        edge_mask = []

        nodes = list(topology.components.keys())

        # Add static edges
        for conn in topology.connections:
            if conn["from"] in nodes and conn["to"] in nodes:
                i = nodes.index(conn["from"])
                j = nodes.index(conn["to"])
                if [i, j] not in list(zip(edge_index[0], edge_index[1], strict=False)):
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_attr.append(np.ones(14))  # Static edge features
                    edge_mask.append(True)  # Static

        # Add dynamic edges based on correlation (K-NN)
        if node_features.shape[0] > 1:
            # Compute pairwise distances
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(node_features)
            np.fill_diagonal(similarity, -1)  # Ignore self

            # For each node, add K nearest neighbors
            for i in range(n_nodes):
                # Get top-k neighbors
                top_k_indices = np.argsort(similarity[i])[-self.k_neighbors :]

                for j in top_k_indices:
                    if similarity[i, j] > self.correlation_threshold:
                        # Check if edge already exists
                        if [i, j] not in list(zip(edge_index[0], edge_index[1], strict=False)):
                            edge_index[0].append(i)
                            edge_index[1].append(j)
                            # Dynamic edge features (correlation-based)
                            edge_attr.append(np.ones(14) * similarity[i, j])
                            edge_mask.append(False)  # Dynamic

        edge_index = np.array(edge_index)
        edge_attr = np.array(edge_attr)
        edge_mask = np.array(edge_mask)

        return edge_index, edge_attr, edge_mask

    async def _load_targets_for_snapshots(
        self, temporal_graphs: list[Data]
    ) -> list[Data]:
        """Load targets for each temporal snapshot.

        Args:
            temporal_graphs: List of graph snapshots

        Returns:
            Graphs with targets attached
        """
        # Mock implementation - real version loads from TimescaleDB
        for graph in temporal_graphs:
            # Graph-level targets
            graph.y_graph_health = torch.rand(1)
            graph.y_graph_degradation = torch.randn(1)
            graph.y_graph_anomaly = torch.zeros(9)  # 9 anomaly classes
            graph.y_graph_rul = torch.rand(1) * 1000

            # Component-level targets
            n_nodes = graph.x.shape[0]
            graph.y_component_health = torch.rand(n_nodes, 1)
            graph.y_component_anomaly = torch.zeros(n_nodes, 9)

        return temporal_graphs
