"""Dynamic Graph Builder for variable topology support.

Builds PyG graphs from arbitrary sensor counts:
- Reads all sensors for equipment from TimescaleDB
- Creates nodes/edges based on actual topology
- No hardcoded N/E assumptions
- Supports multiple equipment types

Python 3.14 Features:
    - Deferred annotations
    - TypeVar for generic flexibility
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

import pandas as pd
import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from src.data.feature_config import FeatureConfig
    from src.data.feature_engineer import FeatureEngineer
    from src.schemas import GraphTopology
    from src.data.timescale_connector import TimescaleConnector

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DynamicGraphBuilder:
    """Build PyG graphs from variable sensor counts.

    Handles:
    - Variable number of sensors per equipment
    - Different equipment types (pump, compressor, motor, etc.)
    - Missing/faulty sensors (replaces with zeros)
    - Automatic node/edge creation based on topology

    Args:
        timescale_connector: Database connector
        feature_engineer: Feature extraction
        feature_config: Configuration (node/edge dimensions)

    Examples:
        >>> builder = DynamicGraphBuilder(
        ...     timescale_connector=connector,
        ...     feature_engineer=engineer,
        ...     feature_config=config
        ... )
        >>>
        >>> # Build graph from TimescaleDB
        >>> graph = await builder.build_from_timescale(
        ...     equipment_id="pump_001",
        ...     topology=topology,
        ...     lookback_minutes=10
        ... )
        >>>
        >>> print(f"Graph: {graph.x.shape[0]} nodes, {graph.edge_attr.shape[0]} edges")
    """

    def __init__(
        self,
        timescale_connector: TimescaleConnector,
        feature_engineer: FeatureEngineer,
        feature_config: FeatureConfig,
    ):
        self.connector = timescale_connector
        self.feature_engineer = feature_engineer
        self.feature_config = feature_config

        logger.info(
            f"DynamicGraphBuilder initialized "
            f"(edge_in_dim={feature_config.edge_in_dim})"
        )

    async def build_from_timescale(
        self,
        equipment_id: str,
        topology: GraphTopology,
        lookback_minutes: int = 10,
    ) -> Data:
        """Build graph from TimescaleDB data.

        Args:
            equipment_id: Equipment identifier
            topology: Equipment topology definition
            lookback_minutes: How far back to read data

        Returns:
            graph: PyG Data object with variable N/E

        Raises:
            ValueError: If topology invalid or no sensors found
            RuntimeError: If database error
        """
        # 1. Read sensors from TimescaleDB
        sensor_data = await self.connector.read_sensor_data(
            equipment_id=equipment_id, lookback_minutes=lookback_minutes
        )

        if sensor_data is None or sensor_data.empty:
            msg = f"No sensor data found for {equipment_id}"
            raise ValueError(msg)

        # 2. Validate sensors match topology
        actual_sensors = set(sensor_data.columns)
        expected_sensors = set(topology.sensor_ids)

        missing = expected_sensors - actual_sensors
        extra = actual_sensors - expected_sensors

        if missing:
            logger.warning(
                f"Missing sensors for {equipment_id}: {missing}. "
                f"Will use zeros for padding."
            )

        if extra:
            logger.warning(
                f"Extra sensors for {equipment_id}: {extra}. "
                f"Will ignore them."
            )

        # 3. Create node features (one per sensor)
        x = self._create_node_features(
            sensor_data=sensor_data,
            topology=topology,
            equipment_id=equipment_id,
        )

        # 4. Create edges based on topology connections
        edge_index = self._create_edge_index(topology=topology)

        # 5. Create edge features
        edge_attr = self._create_edge_features(
            sensor_data=sensor_data,
            edge_index=edge_index,
            topology=topology,
        )

        # 6. Create graph
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            equipment_id=equipment_id,
            topology_id=topology.topology_id,
        )

        logger.info(
            f"Built graph for {equipment_id}: "
            f"{graph.x.shape[0]} nodes, {graph.edge_attr.shape[0]} edges"
        )

        return graph

    def _create_node_features(
        self,
        sensor_data: pd.DataFrame,
        topology: GraphTopology,
        equipment_id: str,
    ) -> torch.Tensor:
        """Create node feature matrix from sensor data.

        Args:
            sensor_data: Time series sensor data (T, S) where S = num sensors
            topology: Equipment topology
            equipment_id: For logging

        Returns:
            x: Node features (N, in_channels) where N = num sensors
        """
        # Get expected sensors from topology
        expected_sensors = topology.sensor_ids
        num_sensors = len(expected_sensors)

        # Initialize node features (N, in_channels)
        x_list = []

        for sensor_id in expected_sensors:
            if sensor_id in sensor_data.columns:
                # Extract sensor time series
                sensor_ts = sensor_data[sensor_id].values  # (T,)

                # Engineer features from time series
                features = self.feature_engineer.extract_all_features(
                    time_series=sensor_ts
                )

                x_list.append(features)
            else:
                # Sensor missing - use zeros
                logger.warning(
                    f"Sensor {sensor_id} missing for {equipment_id}, "
                    f"using zeros"
                )
                features = torch.zeros(self.feature_config.total_features_per_sensor)
                x_list.append(features)

        # Stack all node features
        x = torch.stack(x_list, dim=0)  # (N, in_channels)

        assert x.shape[0] == num_sensors, (
            f"Node count mismatch: {x.shape[0]} vs {num_sensors}"
        )
        assert x.shape[1] == self.feature_config.total_features_per_sensor, (
            f"Feature dim mismatch: {x.shape[1]} vs "
            f"{self.feature_config.total_features_per_sensor}"
        )

        return x

    def _create_edge_index(
        self,
        topology: GraphTopology,
    ) -> torch.Tensor:
        """Create edge connectivity from topology.

        Args:
            topology: Equipment topology with sensor connections

        Returns:
            edge_index: Edge connectivity (2, E)

        Examples:
            >>> # Topology defines connections:
            >>> # "pump_1" <-> "valve_1"
            >>> # "valve_1" <-> "reservoir"
            >>>
            >>> edge_index = builder._create_edge_index(topology)
            >>> print(edge_index.shape)  # (2, 2) - two edges
        """
        # Build sensor ID to node index mapping
        sensor_to_node = {sid: i for i, sid in enumerate(topology.sensor_ids)}

        # Extract edges from topology connections
        edge_list = []

        for connection in topology.connections:
            # connection: {"from": "sensor_1", "to": "sensor_2"}
            from_sensor = connection.get("from") or connection.get("source")
            to_sensor = connection.get("to") or connection.get("target")

            if from_sensor in sensor_to_node and to_sensor in sensor_to_node:
                from_idx = sensor_to_node[from_sensor]
                to_idx = sensor_to_node[to_sensor]

                # Add bidirectional edge
                edge_list.append([from_idx, to_idx])
                edge_list.append([to_idx, from_idx])
            else:
                logger.warning(
                    f"Connection {from_sensor} <-> {to_sensor} "
                    f"references unknown sensors, skipping"
                )

        if not edge_list:
            msg = f"No valid edges found for topology {topology.topology_id}"
            raise ValueError(msg)

        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # (2, E)

        return edge_index

    def _create_edge_features(
        self,
        sensor_data: pd.DataFrame,
        edge_index: torch.Tensor,
        topology: GraphTopology,
    ) -> torch.Tensor:
        """Create edge feature matrix.

        Args:
            sensor_data: Sensor time series data
            edge_index: Edge connectivity (2, E)
            topology: Equipment topology

        Returns:
            edge_attr: Edge features (E, edge_in_dim)
        """
        sensor_to_node = {sid: i for i, sid in enumerate(topology.sensor_ids)}
        num_edges = edge_index.shape[1]

        edge_attr_list = []

        for edge_idx in range(num_edges):
            from_node = edge_index[0, edge_idx].item()
            to_node = edge_index[1, edge_idx].item()

            # Get sensor IDs for this edge
            from_sensor = topology.sensor_ids[from_node]
            to_sensor = topology.sensor_ids[to_node]

            # Compute edge features (correlation, causality, etc.)
            edge_features = self._compute_edge_features(
                from_sensor=from_sensor,
                to_sensor=to_sensor,
                sensor_data=sensor_data,
            )

            edge_attr_list.append(edge_features)

        # Stack all edge features
        edge_attr = torch.stack(edge_attr_list, dim=0)  # (E, edge_in_dim)

        assert edge_attr.shape[0] == num_edges, (
            f"Edge count mismatch: {edge_attr.shape[0]} vs {num_edges}"
        )
        assert edge_attr.shape[1] == self.feature_config.edge_in_dim, (
            f"Edge feature dim mismatch: {edge_attr.shape[1]} vs "
            f"{self.feature_config.edge_in_dim}"
        )

        return edge_attr

    def _compute_edge_features(
        self, from_sensor: str, to_sensor: str, sensor_data: pd.DataFrame
    ) -> torch.Tensor:
        """Compute edge features from sensor pair.

        Args:
            from_sensor: Source sensor ID
            to_sensor: Target sensor ID
            sensor_data: Sensor time series

        Returns:
            features: Edge features (edge_in_dim,)

        Edge features include:
        - Static: distance, type compatibility, expected flow direction
        - Dynamic: correlation, cross-correlation, causality, phase shift
        """
        features = []

        # Static features (8D)
        # 1. Distance (normalized)
        distance = 0.5  # TODO: Get from topology metadata
        features.append(distance)

        # 2. Flow direction (0=from->to, 1=bidirectional)
        flow_dir = 0.0  # TODO: Get from topology
        features.append(flow_dir)

        # 3. Component type compatibility (0=incompatible, 1=compatible)
        type_compat = 1.0  # TODO: Get from topology
        features.append(type_compat)

        # 4-8. Placeholder static features
        features.extend([0.0] * 5)

        # Dynamic features (6D if edge_in_dim=14)
        if self.feature_config.edge_in_dim >= 14:
            # 9. Correlation
            if from_sensor in sensor_data.columns and to_sensor in sensor_data.columns:
                corr = float(
                    sensor_data[from_sensor].corr(sensor_data[to_sensor])
                )
            else:
                corr = 0.0
            features.append(corr)

            # 10. Cross-correlation max
            # TODO: Compute actual cross-correlation
            features.append(0.0)

            # 11. Time lag (in steps)
            features.append(0.0)

            # 12. Phase shift (radians)
            features.append(0.0)

            # 13. Causality score (Granger)
            features.append(0.0)

            # 14. Mutual information
            features.append(0.0)

        # Pad to edge_in_dim if needed
        while len(features) < self.feature_config.edge_in_dim:
            features.append(0.0)

        # Truncate if needed
        features = features[: self.feature_config.edge_in_dim]

        return torch.tensor(features, dtype=torch.float32)

    def validate_graph(self, graph: Data, topology: GraphTopology) -> bool:
        """Validate graph structure.

        Args:
            graph: PyG Data object
            topology: Expected topology

        Returns:
            valid: True if graph is valid
        """
        # Check node count
        if graph.x.shape[0] != len(topology.sensor_ids):
            logger.error(
                f"Node count mismatch: {graph.x.shape[0]} vs "
                f"{len(topology.sensor_ids)}"
            )
            return False

        # Check node features
        if graph.x.shape[1] != self.feature_config.total_features_per_sensor:
            logger.error(
                f"Node feature dim mismatch: {graph.x.shape[1]} vs "
                f"{self.feature_config.total_features_per_sensor}"
            )
            return False

        # Check edge count > 0
        if graph.edge_index.shape[1] == 0:
            logger.error("Graph has no edges")
            return False

        # Check edge features
        if graph.edge_attr.shape[1] != self.feature_config.edge_in_dim:
            logger.error(
                f"Edge feature dim mismatch: {graph.edge_attr.shape[1]} vs "
                f"{self.feature_config.edge_in_dim}"
            )
            return False

        logger.info(
            f"Graph validation passed: "
            f"{graph.x.shape[0]} nodes, {graph.edge_attr.shape[0]} edges"
        )
        return True
