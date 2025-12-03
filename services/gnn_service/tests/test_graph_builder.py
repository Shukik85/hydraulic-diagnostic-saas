"""Tests for GraphBuilder module.

Tests PyTorch Geometric graph construction with 14D edge features:
- Static features (8D)
- Dynamic features (6D)
- Validation and error handling

Author: GNN Service Team
Python: 3.14+
"""

import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.graph_builder import GraphBuilder
from src.schemas import (
    GraphTopology,
    EquipmentMetadata,
    ComponentSpec,
    EdgeSpec,
    ComponentType,
    EdgeType
)
from src.schemas.graph import EdgeMaterial
from src.schemas.requests import ComponentSensorReading
from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer


class TestGraphBuilder:
    """Test GraphBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create GraphBuilder instance."""
        feature_config = FeatureConfig()
        feature_engineer = FeatureEngineer(feature_config)
        
        return GraphBuilder(
            feature_engineer=feature_engineer,
            feature_config=feature_config,
            use_dynamic_features=True
        )
    
    @pytest.fixture
    def builder_no_dynamic(self):
        """Create GraphBuilder without dynamic features (backward compatible)."""
        feature_config = FeatureConfig()
        feature_engineer = FeatureEngineer(feature_config)
        
        return GraphBuilder(
            feature_engineer=feature_engineer,
            feature_config=feature_config,
            use_dynamic_features=False
        )
    
    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data DataFrame."""
        n_timesteps = 100
        
        data = {
            "pressure_pump_1": np.random.normal(150, 5, n_timesteps),
            "temperature_pump_1": np.random.normal(65, 3, n_timesteps),
            "vibration_pump_1": np.random.normal(0.8, 0.1, n_timesteps),
            "pressure_valve_1": np.random.normal(148, 4, n_timesteps),
            "temperature_valve_1": np.random.normal(64, 2, n_timesteps),
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def simple_topology(self):
        """Create simple topology with 2 components and 1 edge."""
        components = {
            "pump_1": ComponentSpec(
                component_id="pump_1",
                component_type=ComponentType.PUMP,
                manufacturer="TestCo",
                model="P1000"
            ),
            "valve_1": ComponentSpec(
                component_id="valve_1",
                component_type=ComponentType.VALVE,
                manufacturer="TestCo",
                model="V500"
            )
        }
        
        edges = [
            EdgeSpec(
                source_id="pump_1",
                target_id="valve_1",
                edge_type=EdgeType.PIPE,
                diameter_mm=25.0,
                length_m=5.0,
                material=EdgeMaterial.STEEL,
                pressure_rating_bar=350.0,
                install_date=datetime.now() - timedelta(days=365)
            )
        ]
        
        return GraphTopology(
            equipment_id="test_equipment",
            components=components,
            edges=edges
        )
    
    @pytest.fixture
    def sample_sensor_readings(self):
        """Create sample sensor readings for dynamic features."""
        return {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0,
                vibration_g=0.8,
                rpm=1450
            ),
            "valve_1": ComponentSensorReading(
                pressure_bar=148.0,
                temperature_c=64.0,
                vibration_g=0.3
            )
        }
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample equipment metadata."""
        return EquipmentMetadata(
            equipment_id="test_equipment",
            equipment_type="hydraulic_system",
            manufacturer="TestCo",
            model="HS-1000",
            installation_date=datetime.now() - timedelta(days=730)
        )


class TestStaticEdgeFeatures:
    """Test static edge feature computation (8D)."""
    
    @pytest.fixture
    def builder(self):
        return GraphBuilder()
    
    @pytest.fixture
    def edge_spec(self):
        return EdgeSpec(
            source_id="pump_1",
            target_id="valve_1",
            edge_type=EdgeType.PIPE,
            diameter_mm=25.0,
            length_m=5.0,
            material=EdgeMaterial.STEEL,
            pressure_rating_bar=350.0
        )
    
    def test_static_features_dimension(self, builder, edge_spec):
        """Test static features have dimension 8."""
        features = builder.build_edge_features_static(edge_spec)
        
        assert features.shape == (8,)
    
    def test_static_features_normalized(self, builder, edge_spec):
        """Test static features are normalized."""
        features = builder.build_edge_features_static(edge_spec)
        
        # Most features should be in reasonable range
        assert np.all(features >= 0)  # All should be non-negative
        assert np.all(features <= 5)  # Should not be extreme
    
    def test_static_features_material_encoding(self, builder, edge_spec):
        """Test material one-hot encoding."""
        features = builder.build_edge_features_static(edge_spec)
        
        # Last 3 features are material encoding
        material_encoding = features[-3:]
        
        # Should be one-hot (exactly one is 1, others are 0)
        assert np.sum(material_encoding) == 1.0
        assert np.all((material_encoding == 0) | (material_encoding == 1))
    
    def test_static_features_different_materials(self, builder):
        """Test different materials have different encodings."""
        edge_steel = EdgeSpec(
            source_id="a", target_id="b", edge_type=EdgeType.PIPE,
            diameter_mm=25.0, length_m=5.0, material=EdgeMaterial.STEEL
        )
        edge_rubber = EdgeSpec(
            source_id="a", target_id="b", edge_type=EdgeType.HOSE,
            diameter_mm=25.0, length_m=5.0, material=EdgeMaterial.RUBBER
        )
        
        features_steel = builder.build_edge_features_static(edge_steel)
        features_rubber = builder.build_edge_features_static(edge_rubber)
        
        # Material encodings should differ
        assert not np.array_equal(features_steel[-3:], features_rubber[-3:])


class TestDynamicEdgeFeatures:
    """Test dynamic edge feature computation (6D)."""
    
    @pytest.fixture
    def builder(self):
        return GraphBuilder(use_dynamic_features=True)
    
    @pytest.fixture
    def edge_spec(self):
        return EdgeSpec(
            source_id="pump_1",
            target_id="valve_1",
            edge_type=EdgeType.PIPE,
            diameter_mm=25.0,
            length_m=5.0,
            material=EdgeMaterial.STEEL,
            install_date=datetime.now() - timedelta(days=365)
        )
    
    @pytest.fixture
    def sensor_readings(self):
        return {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0,
                vibration_g=0.8
            ),
            "valve_1": ComponentSensorReading(
                pressure_bar=148.0,
                temperature_c=64.0,
                vibration_g=0.3
            )
        }
    
    def test_dynamic_features_dimension(self, builder, edge_spec, sensor_readings):
        """Test dynamic features have dimension 6."""
        features = builder.build_edge_features_dynamic(
            edge_spec,
            sensor_readings,
            datetime.now()
        )
        
        assert features.shape == (6,)
    
    def test_dynamic_features_normalized(self, builder, edge_spec, sensor_readings):
        """Test dynamic features are normalized."""
        features = builder.build_edge_features_dynamic(
            edge_spec,
            sensor_readings,
            datetime.now()
        )
        
        # Check reasonable range (most should be in [-5, 5] or [0, 1])
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_dynamic_features_missing_component(self, builder, edge_spec):
        """Test error handling for missing component sensors."""
        sensor_readings = {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0
            )
            # valve_1 missing
        }
        
        # Should not raise, returns defaults instead
        features = builder.build_edge_features_dynamic(
            edge_spec,
            sensor_readings,
            datetime.now()
        )
        
        # Should have defaults (mostly zeros)
        assert features.shape == (6,)


class TestCompleteEdgeFeatures:
    """Test complete edge feature building (14D)."""
    
    @pytest.fixture
    def builder(self):
        return GraphBuilder(use_dynamic_features=True)
    
    @pytest.fixture
    def edge_spec(self):
        return EdgeSpec(
            source_id="pump_1",
            target_id="valve_1",
            edge_type=EdgeType.PIPE,
            diameter_mm=25.0,
            length_m=5.0,
            material=EdgeMaterial.STEEL,
            install_date=datetime.now() - timedelta(days=365)
        )
    
    @pytest.fixture
    def sensor_readings(self):
        return {
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0,
                vibration_g=0.8
            ),
            "valve_1": ComponentSensorReading(
                pressure_bar=148.0,
                temperature_c=64.0,
                vibration_g=0.3
            )
        }
    
    def test_edge_features_dimension_with_dynamic(self, builder, edge_spec, sensor_readings):
        """Test edge features have dimension 14 with dynamic features."""
        features = builder.build_edge_features(
            edge_spec,
            sensor_readings,
            datetime.now()
        )
        
        assert features.shape == torch.Size([14])
    
    def test_edge_features_dimension_without_dynamic(self, edge_spec):
        """Test edge features have dimension 14 (zeros for dynamic) without sensor data."""
        builder = GraphBuilder(use_dynamic_features=True)
        
        # No sensor readings provided
        features = builder.build_edge_features(edge_spec)
        
        # Should still be 14D (with zeros for dynamic)
        assert features.shape == torch.Size([14])
        
        # Last 6 should be zeros
        assert torch.all(features[8:] == 0)
    
    def test_edge_features_backward_compatible(self, edge_spec):
        """Test backward compatibility mode (use_dynamic_features=False)."""
        builder = GraphBuilder(use_dynamic_features=False)
        
        features = builder.build_edge_features(edge_spec)
        
        # Should be 14D with zeros for dynamic
        assert features.shape == torch.Size([14])
        assert torch.all(features[8:] == 0)


class TestGraphConstruction:
    """Test complete graph construction."""
    
    def test_build_graph_with_dynamic(self, builder, sample_sensor_data,
                                     simple_topology, sample_sensor_readings,
                                     sample_metadata):
        """Test building graph with dynamic edge features."""
        graph = builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=simple_topology,
            metadata=sample_metadata,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        # Check graph structure
        assert graph.num_nodes == 2
        assert graph.num_edges == 1
        
        # Check feature dimensions
        assert graph.x.shape[0] == 2  # 2 nodes
        assert graph.edge_attr.shape == torch.Size([1, 14])  # 1 edge, 14 features
    
    def test_build_graph_without_dynamic(self, builder_no_dynamic, sample_sensor_data,
                                        simple_topology, sample_metadata):
        """Test building graph without dynamic features (backward compatible)."""
        graph = builder_no_dynamic.build_graph(
            sensor_data=sample_sensor_data,
            topology=simple_topology,
            metadata=sample_metadata
        )
        
        # Check graph structure
        assert graph.num_nodes == 2
        assert graph.num_edges == 1
        
        # Check edge features are 14D (with zeros)
        assert graph.edge_attr.shape == torch.Size([1, 14])
        
        # Last 6 should be zeros
        assert torch.all(graph.edge_attr[0, 8:] == 0)
    
    def test_validate_graph_valid(self, builder, sample_sensor_data,
                                  simple_topology, sample_metadata):
        """Test graph validation passes for valid graph."""
        graph = builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=simple_topology,
            metadata=sample_metadata
        )
        
        assert builder.validate_graph(graph)
    
    def test_validate_graph_edge_dimension(self, builder):
        """Test validation checks edge dimension is 14."""
        from torch_geometric.data import Data
        
        # Create graph with wrong edge dimension
        graph = Data(
            x=torch.randn(2, 10),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.randn(1, 8)  # Wrong: 8D instead of 14D
        )
        
        # Should fail validation
        assert not builder.validate_graph(graph)
    
    def test_validate_graph_no_nan(self, builder, sample_sensor_data,
                                   simple_topology, sample_metadata):
        """Test validation checks for NaN values."""
        graph = builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=simple_topology,
            metadata=sample_metadata
        )
        
        # Should not contain NaN
        assert not torch.isnan(graph.x).any()
        assert not torch.isnan(graph.edge_attr).any()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_topology(self, builder, sample_sensor_data, sample_metadata):
        """Test handling of empty topology."""
        topology = GraphTopology(
            equipment_id="test",
            components={},
            edges=[]
        )
        
        with pytest.raises(ValueError, match="No components"):
            builder.build_graph(
                sensor_data=sample_sensor_data,
                topology=topology,
                metadata=sample_metadata
            )
    
    def test_missing_sensor_columns(self, builder, simple_topology, sample_metadata):
        """Test handling of missing sensor columns."""
        # Create DataFrame with no matching columns
        sensor_data = pd.DataFrame({
            "random_column": np.random.randn(100)
        })
        
        # Should not crash, will use default features
        graph = builder.build_graph(
            sensor_data=sensor_data,
            topology=simple_topology,
            metadata=sample_metadata
        )
        
        assert graph.num_nodes == 2
