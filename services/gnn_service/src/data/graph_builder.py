"""Graph construction from sensor data and metadata.

Builds PyTorch Geometric graphs:
- Nodes: hydraulic components
- Edges: connections (pipes, hoses)
- Node features: sensor statistics
- Edge features: physical properties

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import logging
from typing import Any

from src.schemas import (
    GraphTopology,
    EquipmentMetadata,
    ComponentSpec,
    EdgeSpec,
    ComponentType,
    EdgeType
)
from src.data.feature_engineer import FeatureEngineer
from src.data.feature_config import FeatureConfig

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build PyG graphs из sensor data и metadata.
    
    Процесс:
    1. Extract component-level features from sensor data
    2. Build node feature matrix [N, F]
    3. Construct edge_index from topology [2, E]
    4. Compute edge features from EdgeSpec [E, F_edge]
    5. Create PyG Data object
    
    Args:
        feature_engineer: FeatureEngineer instance
        feature_config: FeatureConfig instance
    
    Examples:
        >>> builder = GraphBuilder(feature_engineer, feature_config)
        >>> 
        >>> graph = builder.build_graph(
        ...     sensor_data=sensor_df,
        ...     topology=topology,
        ...     metadata=metadata
        ... )
        >>> 
        >>> graph.x.shape  # [num_components, features]
        >>> graph.edge_index.shape  # [2, num_edges]
    """
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer | None = None,
        feature_config: FeatureConfig | None = None
    ):
        self.feature_engineer = feature_engineer or FeatureEngineer(
            FeatureConfig()
        )
        self.feature_config = feature_config or FeatureConfig()
    
    def build_component_features(
        self,
        component_id: str,
        sensor_data: pd.DataFrame,
        component_spec: ComponentSpec
    ) -> torch.Tensor:
        """Построить features для одного component.
        
        Args:
            component_id: Component identifier
            sensor_data: DataFrame с sensor readings [T, sensors]
            component_spec: ComponentSpec с metadata
        
        Returns:
            features: Tensor [F]
        
        Examples:
            >>> sensor_df = pd.DataFrame({
            ...     "pressure_pump_main": [...],
            ...     "temperature_pump_main": [...]
            ... })
            >>> features = builder.build_component_features(
            ...     "pump_main",
            ...     sensor_df,
            ...     component_spec
            ... )
        """
        # Filter sensors для этого component
        component_cols = [
            col for col in sensor_data.columns
            if component_id in col
        ]
        
        if not component_cols:
            logger.warning(f"No sensors found for component {component_id}")
            # Return zeros
            dummy_features = np.zeros(
                self.feature_config.total_features_per_sensor,
                dtype=np.float32
            )
            return torch.from_numpy(dummy_features)
        
        component_data = sensor_data[component_cols]
        
        # Extract features
        features = self.feature_engineer.extract_all_features(component_data)
        
        return torch.from_numpy(features)
    
    def build_edge_features(
        self,
        edge_spec: EdgeSpec
    ) -> torch.Tensor:
        """Построить edge features из EdgeSpec.
        
        Features (8D):
        - diameter_mm (normalized)
        - length_m (normalized)
        - cross_section_area (computed)
        - pressure_loss_coeff (computed from diameter/length)
        - pressure_rating_bar (normalized)
        - one-hot material (3D: steel, rubber, composite)
        
        Args:
            edge_spec: EdgeSpec with physical properties
        
        Returns:
            features: Tensor [8]
        
        Examples:
            >>> edge_spec = EdgeSpec(
            ...     source_id="pump",
            ...     target_id="valve",
            ...     edge_type=EdgeType.PIPE,
            ...     diameter_mm=16.0,
            ...     length_m=2.5,
            ...     pressure_rating_bar=350
            ... )
            >>> features = builder.build_edge_features(edge_spec)
            >>> features.shape  # torch.Size([8])
        """
        features = []
        
        # 1. Diameter (normalized to typical range 6-50mm)
        diameter_norm = edge_spec.diameter_mm / 50.0
        features.append(diameter_norm)
        
        # 2. Length (normalized to typical range 0.1-10m)
        length_norm = edge_spec.length_m / 10.0
        features.append(length_norm)
        
        # 3. Cross-section area (computed)
        radius_m = (edge_spec.diameter_mm / 1000.0) / 2.0
        area = np.pi * radius_m ** 2
        area_norm = area / 0.002  # Normalize to typical max area
        features.append(area_norm)
        
        # 4. Pressure loss coefficient (Darcy-Weisbach approx)
        # Loss ∝ length / diameter^4
        pressure_loss_coeff = edge_spec.length_m / (edge_spec.diameter_mm ** 4 + 1e-6)
        pressure_loss_coeff_norm = np.clip(pressure_loss_coeff * 1000, 0, 1)  # Normalize
        features.append(pressure_loss_coeff_norm)
        
        # 5. Pressure rating (normalized to typical 100-400 bar)
        if edge_spec.pressure_rating_bar is not None:
            rating_norm = edge_spec.pressure_rating_bar / 400.0
            features.append(rating_norm)
        else:
            features.append(0.5)  # Default mid-range
        
        # 6. Material one-hot (3D: steel, rubber, composite)
        material_map = {
            "steel": [1, 0, 0],
            "rubber": [0, 1, 0],
            "composite": [0, 0, 1]
        }
        material = edge_spec.material or "steel"
        material_encoding = material_map.get(material, [1, 0, 0])  # Default steel
        features.extend(material_encoding)
        
        features_array = np.array(features, dtype=np.float32)
        return torch.from_numpy(features_array)
    
    def build_graph(
        self,
        sensor_data: pd.DataFrame,
        topology: GraphTopology,
        metadata: EquipmentMetadata
    ) -> Data:
        """Построить complete PyG graph.
        
        Args:
            sensor_data: DataFrame с sensor readings [T, sensors]
            topology: GraphTopology с components and edges
            metadata: EquipmentMetadata
        
        Returns:
            graph: PyG Data object
        
        Examples:
            >>> graph = builder.build_graph(sensor_df, topology, metadata)
            >>> 
            >>> # Graph structure:
            >>> graph.x          # [N, F] - node features
            >>> graph.edge_index # [2, E] - connectivity
            >>> graph.edge_attr  # [E, F_edge] - edge features
        """
        # 1. Build node features
        node_features_list = []
        component_id_to_idx = {}  # Map component_id -> node index
        
        for idx, (comp_id, comp_spec) in enumerate(topology.components.items()):
            features = self.build_component_features(
                component_id=comp_id,
                sensor_data=sensor_data,
                component_spec=comp_spec
            )
            node_features_list.append(features)
            component_id_to_idx[comp_id] = idx
        
        # Stack node features
        if not node_features_list:
            raise ValueError("No components in topology")
        
        x = torch.stack(node_features_list)  # [N, F]
        
        # 2. Build edge_index and edge_attr
        edge_index_list = []
        edge_attr_list = []
        
        for edge_spec in topology.edges:
            # Get node indices
            source_idx = component_id_to_idx.get(edge_spec.source_id)
            target_idx = component_id_to_idx.get(edge_spec.target_id)
            
            if source_idx is None or target_idx is None:
                logger.warning(f"Edge references unknown component: {edge_spec}")
                continue
            
            # Add edge
            edge_index_list.append([source_idx, target_idx])
            
            # Add edge features
            edge_features = self.build_edge_features(edge_spec)
            edge_attr_list.append(edge_features)
            
            # Add reverse edge if bidirectional
            if edge_spec.is_bidirectional:
                edge_index_list.append([target_idx, source_idx])
                edge_attr_list.append(edge_features)  # Same features
        
        # Convert to tensors
        if not edge_index_list:
            logger.warning("No edges in topology - creating self-loops")
            # Create self-loops
            edge_index = torch.tensor(
                [[i, i] for i in range(len(node_features_list))],
                dtype=torch.long
            ).t().contiguous()
            edge_attr = torch.zeros(len(node_features_list), 8)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attr_list)
        
        # 3. Create PyG Data
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # 4. Validate
        if not self.validate_graph(graph):
            raise ValueError("Invalid graph structure")
        
        logger.info(
            f"Built graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
            f"node features: {graph.x.shape[1]}, edge features: {graph.edge_attr.shape[1]}"
        )
        
        return graph
    
    def validate_graph(self, data: Data) -> bool:
        """Валидировать PyG graph structure.
        
        Checks:
        - Node features exist and have correct shape
        - Edge index valid (no out-of-bounds indices)
        - Edge attributes match edge count
        - No NaN/inf values
        
        Args:
            data: PyG Data object
        
        Returns:
            valid: True если graph valid
        
        Examples:
            >>> graph = builder.build_graph(...)
            >>> is_valid = builder.validate_graph(graph)
            >>> assert is_valid
        """
        # Check node features
        if data.x is None or data.x.numel() == 0:
            logger.error("Node features are empty")
            return False
        
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            logger.error("Node features contain NaN or inf")
            return False
        
        # Check edge index
        if data.edge_index is None or data.edge_index.numel() == 0:
            logger.warning("Edge index is empty (graph has no edges)")
            # Allow graphs without edges (single component)
        else:
            max_idx = data.edge_index.max().item()
            if max_idx >= data.num_nodes:
                logger.error(f"Edge index out of bounds: {max_idx} >= {data.num_nodes}")
                return False
        
        # Check edge attributes
        if data.edge_attr is not None:
            if data.edge_attr.shape[0] != data.num_edges:
                logger.error(
                    f"Edge attr count mismatch: {data.edge_attr.shape[0]} != {data.num_edges}"
                )
                return False
            
            if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
                logger.error("Edge features contain NaN or inf")
                return False
        
        return True
    
    def get_component_sensor_columns(
        self,
        component_id: str,
        sensor_data: pd.DataFrame
    ) -> list[str]:
        """Найти sensor columns для component.
        
        Convention: sensor columns named as "{sensor_type}_{component_id}"
        Example: "pressure_pump_main", "temperature_valve_01"
        
        Args:
            component_id: Component identifier
            sensor_data: DataFrame с sensor columns
        
        Returns:
            columns: Список column names
        
        Examples:
            >>> cols = builder.get_component_sensor_columns("pump_main", sensor_df)
            >>> # ["pressure_pump_main", "temperature_pump_main", "vibration_pump_main"]
        """
        return [
            col for col in sensor_data.columns
            if component_id in col
        ]
