"""Temporal dataloader for hydraulic diagnostics with GRAPE imputation.

Provides:
- Temporal snapshot creation
- Dynamic edge construction
- GRAPE two-stage imputation (optional)
- Missing data handling
- Target loading from TimescaleDB

Examples:
    >>> loader = TemporalHydraulicDataLoader(
    ...     timescale_connector=connector,
    ...     window_size=3600,
    ...     config={"imputation": {"enabled": True}},
    ... )
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from src.data.timescale_connector import TimescaleConnector
    from src.topology.mock_topology import GraphTopology

logger = logging.getLogger(__name__)


class TemporalHydraulicDataLoader:
    """DataLoader for temporal hydraulic diagnostics with GRAPE imputation.

    Supports optional two-stage imputation for handling missing sensors.

    Examples:
        >>> config = {
        ...     "imputation": {
        ...         "enabled": True,
        ...         "spatial": {"hidden_dim": 128, "use_static_prior": True},
        ...         "temporal": {"hidden_dim": 128},
        ...     }
        ... }
        >>> loader = TemporalHydraulicDataLoader(
        ...     timescale_connector=connector,
        ...     window_size=3600,
        ...     config=config,
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
        config: dict | None = None,
        device: str = "cpu",
    ):
        self.timescale = timescale_connector
        self.window_size = window_size
        self.stride = stride
        self.sequence_length = sequence_length
        self.correlation_threshold = correlation_threshold
        self.k_neighbors = k_neighbors
        self.max_missing_ratio = max_missing_ratio
        self.device = device
        self.config = config or {}

        # Initialize GRAPE imputation if enabled
        self.imputation_enabled = self.config.get("imputation", {}).get("enabled", False)
        self.imputer = None

        if self.imputation_enabled:
            logger.info("🔬 Initializing GRAPE two-stage imputation")
            from src.training.imputation_grape import TwoStageImputer

            imputation_config = self.config["imputation"]
            spatial_config = imputation_config.get("spatial", {})
            temporal_config = imputation_config.get("temporal", {})

            self.imputer = TwoStageImputer(
                feature_dim=34,
                spatial_hidden=spatial_config.get("hidden_dim", 128),
                temporal_hidden=temporal_config.get("hidden_dim", 128),
                spatial_layers=spatial_config.get("num_layers", 2),
                temporal_layers=temporal_config.get("num_layers", 2),
                device=device,
            )
            logger.info("   ✅ GRAPE imputer initialized")
            logger.info(f"   - Spatial hidden: {spatial_config.get('hidden_dim', 128)}")
            logger.info(f"   - Temporal hidden: {temporal_config.get('hidden_dim', 128)}")
            logger.info(f"   - Use static prior: {spatial_config.get('use_static_prior', True)}")

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

        # Apply GRAPE imputation if enabled
        if self.imputation_enabled and len(temporal_graphs) > 0:
            logger.info(f"🔬 Applying GRAPE imputation to {len(temporal_graphs)} snapshots")
            temporal_graphs = self._apply_grape_imputation(temporal_graphs, topology)
            logger.info("   ✅ Imputation complete")

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

        # Convert window_size and stride from seconds to timedelta
        window_td = timedelta(seconds=self.window_size)
        stride_td = timedelta(seconds=self.stride)

        # Generate window starts
        window_starts = []
        current_time = min_time
        while current_time + window_td <= max_time:
            window_starts.append(current_time)
            current_time += stride_td

        # Create snapshots
        snapshots = []
        for window_start in window_starts:
            window_end = window_start + window_td

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
                topology, node_features, mask_nodes
            )

            # Create graph
            # Store timestamp as int64 (epoch seconds) for PyG batching compatibility
            timestamp_epoch = int(window_start.timestamp())
            
            snapshot = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                mask_nodes=torch.tensor(mask_nodes, dtype=torch.bool),
                edge_mask=torch.tensor(edge_mask, dtype=torch.bool),
                timestamp=torch.tensor(timestamp_epoch, dtype=torch.int64),
            )

            snapshots.append(snapshot)

        return snapshots

    def _apply_grape_imputation(
        self, snapshots: list[Data], topology: GraphTopology
    ) -> list[Data]:
        """Apply GRAPE spatial imputation to snapshots.

        Args:
            snapshots: List of graph snapshots
            topology: Graph topology for static edges

        Returns:
            Snapshots with imputed features and confidence scores
        """
        if self.imputer is None:
            logger.warning("Imputer not initialized, skipping imputation")
            return snapshots

        # Get static topology edges
        static_edges = self._get_static_topology_edges(topology)
        static_edges_tensor = torch.tensor(static_edges, dtype=torch.long).to(self.device)

        # Apply spatial imputation to each snapshot
        imputed_snapshots = []
        for snapshot in snapshots:
            # Move to device
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device)
            mask_nodes = snapshot.mask_nodes.to(self.device)

            # Apply spatial imputation
            x_imputed, confidence = self.imputer.spatial_imputer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask_nodes=mask_nodes,
                static_topology=static_edges_tensor if static_edges_tensor.numel() > 0 else None,
            )

            # Update snapshot with imputed data
            snapshot.x = x_imputed.cpu()
            snapshot.confidence = confidence.cpu()

            imputed_snapshots.append(snapshot)

        return imputed_snapshots

    def _get_static_topology_edges(self, topology: GraphTopology) -> np.ndarray:
        """Extract static edges from topology.

        Args:
            topology: Graph topology

        Returns:
            Static edge_index [2, E]
        """
        nodes = list(topology.components.keys())
        # Build node_id -> index mapping once
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
        
        edge_index = [[], []]

        for conn in topology.connections:
            if conn["from"] in node_to_idx and conn["to"] in node_to_idx:
                i = node_to_idx[conn["from"]]
                j = node_to_idx[conn["to"]]
                edge_index[0].append(i)
                edge_index[1].append(j)

        return np.array(edge_index) if edge_index[0] else np.array([[], []])

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
                # TODO: Replace with FeatureEngineer for full 34-dim extraction
                node_features[idx, :4] = [
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                ]
                mask_nodes[idx] = True
        
        # Log feature extraction stats
        n_observed = mask_nodes.sum()
        n_missing = n_nodes - n_observed
        logger.debug(
            f"Extracted features: {n_observed}/{n_nodes} nodes observed, "
            f"{n_missing} missing (only first 4/{n_features} dims populated)"
        )

        return node_features, mask_nodes

    def _construct_dynamic_edges(
        self, 
        topology: GraphTopology, 
        node_features: np.ndarray,
        mask_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct dynamic edges from sensor correlations.

        Args:
            topology: Graph topology (static edges)
            node_features: Node features [N, F]
            mask_nodes: Boolean mask [N] (True=observed)

        Returns:
            edge_index: [2, E]
            edge_attr: [E, 14]
            edge_mask: [E] bool (True=static, False=dynamic)
        """
        n_nodes = len(topology.components)
        nodes = list(topology.components.keys())
        
        # Build node_id -> index mapping once
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}

        # Start with static edges from topology
        edge_set = set()  # Use set of tuples for efficient deduplication
        edge_list = []  # Preserve order for edge_attr/edge_mask alignment
        edge_attr = []
        edge_mask = []

        # Add static edges
        for conn in topology.connections:
            if conn["from"] in node_to_idx and conn["to"] in node_to_idx:
                i = node_to_idx[conn["from"]]
                j = node_to_idx[conn["to"]]
                edge_tuple = (i, j)
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    edge_list.append([i, j])
                    edge_attr.append(np.ones(14))  # Static edge features
                    edge_mask.append(True)  # Static

        # Add dynamic edges based on correlation (K-NN) only for observed nodes
        observed_indices = np.where(mask_nodes)[0]
        
        if len(observed_indices) > 1:
            # Extract features only for observed nodes
            observed_features = node_features[observed_indices]
            
            # Compute pairwise similarity among observed nodes
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(observed_features)
            np.fill_diagonal(similarity, -1)  # Ignore self

            # For each observed node, add K nearest neighbors
            for local_i, global_i in enumerate(observed_indices):
                # Get top-k neighbors (in local observed space)
                top_k_local = np.argsort(similarity[local_i])[-self.k_neighbors :]

                for local_j in top_k_local:
                    global_j = observed_indices[local_j]
                    edge_tuple = (int(global_i), int(global_j))
                    
                    # Check correlation threshold and avoid duplicates
                    if (
                        similarity[local_i, local_j] > self.correlation_threshold
                        and edge_tuple not in edge_set
                    ):
                        edge_set.add(edge_tuple)
                        edge_list.append([int(global_i), int(global_j)])
                        # Dynamic edge features (correlation-based)
                        edge_attr.append(np.ones(14) * similarity[local_i, local_j])
                        edge_mask.append(False)  # Dynamic

        # Convert to numpy arrays
        if edge_list:
            edge_index = np.array(edge_list).T  # [2, E]
            edge_attr_array = np.array(edge_attr)
            edge_mask_array = np.array(edge_mask)
        else:
            # Empty graph (no edges)
            edge_index = np.array([[], []])
            edge_attr_array = np.zeros((0, 14))
            edge_mask_array = np.zeros(0, dtype=bool)
        
        n_static = edge_mask_array.sum() if len(edge_mask_array) > 0 else 0
        n_dynamic = len(edge_mask_array) - n_static
        logger.debug(
            f"Constructed edges: {len(edge_mask_array)} total "
            f"({n_static} static, {n_dynamic} dynamic)"
        )

        return edge_index, edge_attr_array, edge_mask_array

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

            # Add confidence if not present (for non-imputed graphs)
            if not hasattr(graph, "confidence"):
                graph.confidence = torch.ones(n_nodes)

        return temporal_graphs
