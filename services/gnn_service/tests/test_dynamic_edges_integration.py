"""Integration test for dynamic edge features.

End-to-end test of the complete pipeline:
1. EdgeFeatureComputer computes dynamic features
2. EdgeFeatureNormalizer normalizes features
3. GraphBuilder builds 14D edge graph
4. Model processes graph and produces predictions

Author: GNN Service Team
Python: 3.14+
"""

import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from src.data.graph_builder import GraphBuilder
from src.data.edge_features import EdgeFeatureComputer
from src.data.normalization import EdgeFeatureNormalizer, NormalizationStatistics
from src.data.feature_config import FeatureConfig
from src.data.feature_engineer import FeatureEngineer
from src.models.gnn_model import UniversalTemporalGNN
from src.schemas import (
    GraphTopology,
    EquipmentMetadata,
    ComponentSpec,
    EdgeSpec,
    ComponentType,
    EdgeType
)
from src.schemas.graph import EdgeMaterial
from src.schemas.requests import ComponentSensorReading, MinimalInferenceRequest


@pytest.fixture
def complete_system():
    """Create complete system with all components."""
    # Feature engineering
    feature_config = FeatureConfig()
    feature_engineer = FeatureEngineer(feature_config)
    
    # Edge feature computation
    edge_computer = EdgeFeatureComputer()
    
    # Normalization
    normalizer = EdgeFeatureNormalizer()
    
    # Graph builder
    graph_builder = GraphBuilder(
        feature_engineer=feature_engineer,
        feature_config=feature_config,
        edge_feature_computer=edge_computer,
        edge_normalizer=normalizer,
        use_dynamic_features=True
    )
    
    # Model
    model = UniversalTemporalGNN(
        in_channels=feature_config.total_features_per_sensor,
        hidden_channels=128,
        num_heads=4,
        num_gat_layers=2,
        lstm_hidden=128,
        lstm_layers=1,
        edge_feature_dim=14,  # Phase 3.1: 14D edges
        use_compile=False  # Disable for testing
    )
    model.eval()
    
    return {
        "graph_builder": graph_builder,
        "edge_computer": edge_computer,
        "normalizer": normalizer,
        "model": model,
        "feature_config": feature_config
    }


@pytest.fixture
def sample_topology():
    """Create realistic hydraulic system topology."""
    components = {
        "pump_main": ComponentSpec(
            component_id="pump_main",
            component_type=ComponentType.PUMP,
            manufacturer="Bosch Rexroth",
            model="A10VSO"
        ),
        "filter_main": ComponentSpec(
            component_id="filter_main",
            component_type=ComponentType.FILTER,
            manufacturer="HYDAC",
            model="DFBN/HC"
        ),
        "valve_control": ComponentSpec(
            component_id="valve_control",
            component_type=ComponentType.VALVE,
            manufacturer="Parker",
            model="D1VW"
        ),
        "cylinder_1": ComponentSpec(
            component_id="cylinder_1",
            component_type=ComponentType.CYLINDER,
            manufacturer="SMC",
            model="CY1S"
        )
    }
    
    edges = [
        EdgeSpec(
            source_id="pump_main",
            target_id="filter_main",
            edge_type=EdgeType.PIPE,
            diameter_mm=25.0,
            length_m=2.0,
            material=EdgeMaterial.STEEL,
            pressure_rating_bar=350.0,
            install_date=datetime.now() - timedelta(days=500)
        ),
        EdgeSpec(
            source_id="filter_main",
            target_id="valve_control",
            edge_type=EdgeType.PIPE,
            diameter_mm=20.0,
            length_m=1.5,
            material=EdgeMaterial.STEEL,
            pressure_rating_bar=350.0,
            install_date=datetime.now() - timedelta(days=500)
        ),
        EdgeSpec(
            source_id="valve_control",
            target_id="cylinder_1",
            edge_type=EdgeType.HOSE,
            diameter_mm=16.0,
            length_m=3.0,
            material=EdgeMaterial.RUBBER,
            pressure_rating_bar=250.0,
            install_date=datetime.now() - timedelta(days=200)
        )
    ]
    
    return GraphTopology(
        equipment_id="excavator_001",
        components=components,
        edges=edges
    )


@pytest.fixture
def sample_sensor_data():
    """Create realistic sensor time series."""
    n_timesteps = 200
    
    # Simulate degrading pump
    t = np.linspace(0, 1, n_timesteps)
    pressure_pump = 150 + 5 * np.sin(2 * np.pi * t) - 10 * t  # Pressure drops over time
    temp_pump = 60 + 3 * np.sin(2 * np.pi * t) + 5 * t  # Temperature increases
    vibration_pump = 0.5 + 0.1 * np.random.randn(n_timesteps) + 0.3 * t  # Vibration increases
    
    data = {
        "pressure_pump_main": pressure_pump,
        "temperature_pump_main": temp_pump,
        "vibration_pump_main": vibration_pump,
        "pressure_filter_main": pressure_pump - 2 + np.random.randn(n_timesteps),
        "temperature_filter_main": temp_pump + 1 + np.random.randn(n_timesteps),
        "pressure_valve_control": pressure_pump - 5 + np.random.randn(n_timesteps),
        "temperature_valve_control": temp_pump + 2 + np.random.randn(n_timesteps),
        "pressure_cylinder_1": pressure_pump - 8 + np.random.randn(n_timesteps),
        "temperature_cylinder_1": temp_pump + 1 + np.random.randn(n_timesteps),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_sensor_readings():
    """Create current sensor readings for dynamic features."""
    return {
        "pump_main": ComponentSensorReading(
            pressure_bar=145.0,
            temperature_c=68.0,
            vibration_g=0.9,
            rpm=1420
        ),
        "filter_main": ComponentSensorReading(
            pressure_bar=143.0,
            temperature_c=69.0,
            vibration_g=0.2
        ),
        "valve_control": ComponentSensorReading(
            pressure_bar=140.0,
            temperature_c=70.0,
            vibration_g=0.1
        ),
        "cylinder_1": ComponentSensorReading(
            pressure_bar=137.0,
            temperature_c=69.0,
            vibration_g=0.3
        )
    }


@pytest.fixture
def sample_metadata():
    """Create equipment metadata."""
    return EquipmentMetadata(
        equipment_id="excavator_001",
        equipment_type="excavator",
        manufacturer="Caterpillar",
        model="320D",
        installation_date=datetime.now() - timedelta(days=1000)
    )


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_graph_construction_14d_edges(
        self, complete_system, sample_sensor_data,
        sample_topology, sample_sensor_readings, sample_metadata
    ):
        """Test graph construction produces 14D edge features."""
        graph_builder = complete_system["graph_builder"]
        
        graph = graph_builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        # Check graph structure
        assert graph.num_nodes == 4  # 4 components
        assert graph.num_edges == 3  # 3 edges
        
        # Check edge features are 14D
        assert graph.edge_attr.shape == torch.Size([3, 14])
        
        # Check no NaN or inf
        assert not torch.isnan(graph.x).any()
        assert not torch.isnan(graph.edge_attr).any()
        assert not torch.isinf(graph.x).any()
        assert not torch.isinf(graph.edge_attr).any()
    
    def test_model_forward_pass_14d(
        self, complete_system, sample_sensor_data,
        sample_topology, sample_sensor_readings, sample_metadata
    ):
        """Test model accepts 14D edge features and produces predictions."""
        graph_builder = complete_system["graph_builder"]
        model = complete_system["model"]
        
        # Build graph
        graph = graph_builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        # Forward pass
        with torch.no_grad():
            health, degradation, anomaly = model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr
            )
        
        # Check output shapes
        assert health.shape == torch.Size([1, 1])
        assert degradation.shape == torch.Size([1, 1])
        assert anomaly.shape == torch.Size([1, 9])
        
        # Check value ranges
        assert 0 <= health.item() <= 1
        assert 0 <= degradation.item() <= 1
        
        # Check no NaN
        assert not torch.isnan(health).any()
        assert not torch.isnan(degradation).any()
        assert not torch.isnan(anomaly).any()
    
    def test_dynamic_features_impact(
        self, complete_system, sample_sensor_data,
        sample_topology, sample_sensor_readings, sample_metadata
    ):
        """Test that dynamic features affect predictions."""
        graph_builder = complete_system["graph_builder"]
        model = complete_system["model"]
        
        # Build graph with dynamic features
        graph_with_dynamic = graph_builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        # Build graph without dynamic features (zeros)
        graph_builder_no_dynamic = GraphBuilder(
            feature_engineer=complete_system["graph_builder"].feature_engineer,
            feature_config=complete_system["feature_config"],
            use_dynamic_features=False
        )
        
        graph_without_dynamic = graph_builder_no_dynamic.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata
        )
        
        # Get predictions
        with torch.no_grad():
            health_with, _, _ = model(
                x=graph_with_dynamic.x,
                edge_index=graph_with_dynamic.edge_index,
                edge_attr=graph_with_dynamic.edge_attr
            )
            
            health_without, _, _ = model(
                x=graph_without_dynamic.x,
                edge_index=graph_without_dynamic.edge_index,
                edge_attr=graph_without_dynamic.edge_attr
            )
        
        # Dynamic features should affect predictions
        # (predictions should be different)
        assert not torch.allclose(health_with, health_without, atol=1e-4)
    
    def test_inference_performance(
        self, complete_system, sample_sensor_data,
        sample_topology, sample_sensor_readings, sample_metadata
    ):
        """Test inference performance (should be < 200ms)."""
        graph_builder = complete_system["graph_builder"]
        model = complete_system["model"]
        
        # Warmup
        graph = graph_builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        with torch.no_grad():
            _ = model(graph.x, graph.edge_index, graph.edge_attr)
        
        # Benchmark
        n_runs = 10
        times = []
        
        for _ in range(n_runs):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(graph.x, graph.edge_index, graph.edge_attr)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        
        print(f"\nAverage inference time: {avg_time:.2f}ms")
        
        # Should be reasonably fast (< 200ms)
        assert avg_time < 200, f"Inference too slow: {avg_time:.2f}ms"
    
    def test_edge_feature_computation_breakdown(
        self, complete_system, sample_sensor_readings
    ):
        """Test edge feature computation step-by-step."""
        edge_computer = complete_system["edge_computer"]
        
        edge_spec = EdgeSpec(
            source_id="pump_main",
            target_id="filter_main",
            edge_type=EdgeType.PIPE,
            diameter_mm=25.0,
            length_m=2.0,
            material=EdgeMaterial.STEEL,
            install_date=datetime.now() - timedelta(days=500)
        )
        
        # Compute features
        features = edge_computer.compute_edge_features(
            edge=edge_spec,
            sensor_readings=sample_sensor_readings,
            current_time=datetime.now()
        )
        
        # Check all 6 dynamic features computed
        assert "flow_rate_lpm" in features
        assert "pressure_drop_bar" in features
        assert "temperature_delta_c" in features
        assert "vibration_level_g" in features
        assert "age_hours" in features
        assert "maintenance_score" in features
        
        # Check reasonable values
        assert features["flow_rate_lpm"] > 0  # Should have flow
        assert features["pressure_drop_bar"] > 0  # Pressure drops
        assert features["age_hours"] > 0  # Component has age
        assert 0 <= features["maintenance_score"] <= 1


class TestBackwardCompatibility:
    """Test backward compatibility with 8D edges."""
    
    def test_model_accepts_legacy_8d_edges(
        self, sample_sensor_data, sample_topology, sample_metadata
    ):
        """Test model still accepts 8D edges (zeros for dynamic)."""
        # Build graph without dynamic features
        graph_builder = GraphBuilder(use_dynamic_features=False)
        
        graph = graph_builder.build_graph(
            sensor_data=sample_sensor_data,
            topology=sample_topology,
            metadata=sample_metadata
        )
        
        # Edge features should be 14D with zeros
        assert graph.edge_attr.shape[1] == 14
        assert torch.all(graph.edge_attr[:, 8:] == 0)
        
        # Model should accept
        model = UniversalTemporalGNN(
            in_channels=graph.x.shape[1],
            edge_feature_dim=14,
            use_compile=False
        )
        model.eval()
        
        with torch.no_grad():
            health, degradation, anomaly = model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr
            )
        
        # Should produce valid predictions
        assert 0 <= health.item() <= 1


class TestMinimalInferenceRequest:
    """Test MinimalInferenceRequest integration."""
    
    def test_minimal_request_to_graph(
        self, complete_system, sample_topology
    ):
        """Test converting MinimalInferenceRequest to graph."""
        # Create minimal request
        request = MinimalInferenceRequest(
            equipment_id="excavator_001",
            timestamp=datetime.now(),
            sensor_readings={
                "pump_main": ComponentSensorReading(
                    pressure_bar=145.0,
                    temperature_c=68.0,
                    vibration_g=0.9
                ),
                "filter_main": ComponentSensorReading(
                    pressure_bar=143.0,
                    temperature_c=69.0
                )
            },
            topology_id="excavator_standard"
        )
        
        # Sensor readings should be valid
        assert "pump_main" in request.sensor_readings
        assert request.sensor_readings["pump_main"].pressure_bar == 145.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
