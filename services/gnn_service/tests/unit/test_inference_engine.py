"""Unit tests for InferenceEngine.

Tests:
- Single prediction
- Batch prediction
- Preprocessing
- Postprocessing
- Error handling
- GPU optimization
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data, Batch

from src.inference import InferenceEngine, InferenceConfig
from src.data import FeatureConfig
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    GraphTopology,
    ComponentSpec,
    EdgeSpec,
    ComponentType,
    EdgeType
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_topology():
    """Sample GraphTopology."""
    components = {
        "pump_main": ComponentSpec(
            component_id="pump_main",
            component_type=ComponentType.PUMP,
            location_x=0.0,
            location_y=0.0,
            location_z=0.0
        ),
        "valve_control": ComponentSpec(
            component_id="valve_control",
            component_type=ComponentType.VALVE,
            location_x=1.0,
            location_y=0.0,
            location_z=0.0
        )
    }
    
    edges = [
        EdgeSpec(
            source_id="pump_main",
            target_id="valve_control",
            edge_type=EdgeType.PIPE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350,
            material="steel"
        )
    ]
    
    return GraphTopology(components=components, edges=edges)


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data."""
    return pd.DataFrame({
        "pressure_pump_main": np.random.randn(100) * 10 + 150,
        "temperature_pump_main": np.random.randn(100) * 5 + 65,
        "vibration_pump_main": np.random.randn(100) * 0.5 + 2.5
    })


@pytest.fixture
def sample_prediction_request(sample_sensor_data):
    """Sample PredictionRequest."""
    return PredictionRequest(
        equipment_id="exc_001",
        sensor_data=sample_sensor_data.to_dict(orient="list")
    )


@pytest.fixture
def mock_graph():
    """Mock PyG graph."""
    return Data(
        x=torch.randn(2, 34),  # 2 nodes, 34 features
        edge_index=torch.tensor([[0], [1]]),  # 1 edge
        edge_attr=torch.randn(1, 8)  # 8D edge features
    )


@pytest.fixture
def mock_model():
    """Mock model."""
    model = Mock()
    model.parameters = Mock(return_value=[torch.randn(10, 10)])
    
    # Mock forward pass
    def forward_side_effect(x, edge_index, edge_attr, batch):
        batch_size = batch.max().item() + 1
        health = torch.rand(batch_size, 1) * 0.5 + 0.5  # 0.5-1.0
        degradation = torch.rand(batch_size, 1) * 0.2  # 0.0-0.2
        anomaly = torch.randn(batch_size, 9)  # Logits
        return health, degradation, anomaly
    
    model.side_effect = forward_side_effect
    return model


@pytest.fixture
def mock_model_manager(mock_model, tmp_path):
    """Mock ModelManager."""
    manager = Mock()
    
    # Create dummy checkpoint
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({}, ckpt_path)
    
    manager.load_model = Mock(return_value=mock_model)
    manager.warmup = Mock()
    manager.get_model_info = Mock(return_value={
        "device": "cpu",
        "num_parameters": 2500000
    })
    
    return manager


# ==================== TESTS ====================

class TestInferenceEngineInit:
    """Test InferenceEngine initialization."""
    
    @patch('src.inference.inference_engine.ModelManager')
    def test_engine_initialization(self, mock_manager_class, tmp_path):
        """Test engine initializes correctly."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu",
            batch_size=32
        )
        
        # Initialize
        engine = InferenceEngine(config=config)
        
        # Assertions
        assert engine.config == config
        assert engine.model is mock_model
        mock_manager.load_model.assert_called_once()
        mock_manager.warmup.assert_called_once()


class TestSinglePrediction:
    """Test single prediction flow."""
    
    @patch('src.inference.inference_engine.ModelManager')
    @patch('src.inference.inference_engine.GraphBuilder')
    async def test_predict_success(
        self,
        mock_builder_class,
        mock_manager_class,
        sample_prediction_request,
        sample_topology,
        mock_graph,
        tmp_path
    ):
        """Test successful single prediction."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        # Mock ModelManager
        mock_manager = Mock()
        mock_model = Mock()
        
        def forward_mock(x, edge_index, edge_attr, batch):
            return (
                torch.tensor([[0.87]]),  # health
                torch.tensor([[0.12]]),  # degradation
                torch.randn(1, 9)  # anomaly
            )
        
        mock_model.side_effect = forward_mock
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock GraphBuilder
        mock_builder = Mock()
        mock_builder.build_graph = Mock(return_value=mock_graph)
        mock_builder_class.return_value = mock_builder
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu"
        )
        
        engine = InferenceEngine(config=config)
        
        # Predict
        response = await engine.predict(
            request=sample_prediction_request,
            topology=sample_topology
        )
        
        # Assertions
        assert isinstance(response, PredictionResponse)
        assert response.equipment_id == "exc_001"
        assert 0 <= response.health.score <= 1
        assert 0 <= response.degradation.rate <= 1
        assert len(response.anomaly.predictions) == 9
        assert response.inference_time_ms > 0
    
    @patch('src.inference.inference_engine.ModelManager')
    @patch('src.inference.inference_engine.GraphBuilder')
    async def test_predict_model_error(
        self,
        mock_builder_class,
        mock_manager_class,
        sample_prediction_request,
        sample_topology,
        mock_graph,
        tmp_path
    ):
        """Test prediction handles model errors."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        # Mock ModelManager
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.side_effect = RuntimeError("Model forward failed")
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock GraphBuilder
        mock_builder = Mock()
        mock_builder.build_graph = Mock(return_value=mock_graph)
        mock_builder_class.return_value = mock_builder
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu"
        )
        
        engine = InferenceEngine(config=config)
        
        # Should raise
        with pytest.raises(Exception):
            await engine.predict(
                request=sample_prediction_request,
                topology=sample_topology
            )


class TestBatchPrediction:
    """Test batch prediction flow."""
    
    @patch('src.inference.inference_engine.ModelManager')
    @patch('src.inference.inference_engine.GraphBuilder')
    async def test_predict_batch_success(
        self,
        mock_builder_class,
        mock_manager_class,
        sample_topology,
        mock_graph,
        sample_sensor_data,
        tmp_path
    ):
        """Test successful batch prediction."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        # Create batch requests
        requests = [
            PredictionRequest(
                equipment_id=f"exc_{i:03d}",
                sensor_data=sample_sensor_data.to_dict(orient="list")
            )
            for i in range(3)
        ]
        
        # Mock ModelManager
        mock_manager = Mock()
        mock_model = Mock()
        
        def forward_mock(x, edge_index, edge_attr, batch):
            batch_size = batch.max().item() + 1
            return (
                torch.rand(batch_size, 1),
                torch.rand(batch_size, 1),
                torch.randn(batch_size, 9)
            )
        
        mock_model.side_effect = forward_mock
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock GraphBuilder
        mock_builder = Mock()
        mock_builder.build_graph = Mock(return_value=mock_graph)
        mock_builder_class.return_value = mock_builder
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu"
        )
        
        engine = InferenceEngine(config=config)
        
        # Predict batch
        responses = await engine.predict_batch(
            requests=requests,
            topology=sample_topology
        )
        
        # Assertions
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.equipment_id == f"exc_{i:03d}"
            assert isinstance(response, PredictionResponse)


class TestPreprocessing:
    """Test preprocessing logic."""
    
    @patch('src.inference.inference_engine.ModelManager')
    @patch('src.inference.inference_engine.GraphBuilder')
    def test_preprocess_builds_graph(
        self,
        mock_builder_class,
        mock_manager_class,
        sample_prediction_request,
        sample_topology,
        mock_graph,
        tmp_path
    ):
        """Test _preprocess calls GraphBuilder."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        mock_builder = Mock()
        mock_builder.build_graph = Mock(return_value=mock_graph)
        mock_builder_class.return_value = mock_builder
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu"
        )
        
        engine = InferenceEngine(config=config)
        
        # Call preprocess
        graph = engine._preprocess(
            request=sample_prediction_request,
            topology=sample_topology
        )
        
        # Assertions
        assert graph is mock_graph
        mock_builder.build_graph.assert_called_once()


class TestPostprocessing:
    """Test postprocessing logic."""
    
    @patch('src.inference.inference_engine.ModelManager')
    def test_postprocess_formats_response(
        self,
        mock_manager_class,
        sample_prediction_request,
        tmp_path
    ):
        """Test _postprocess formats PredictionResponse."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager_class.return_value = mock_manager
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu"
        )
        
        engine = InferenceEngine(config=config)
        
        # Mock outputs
        health = torch.tensor([[0.87]])
        degradation = torch.tensor([[0.12]])
        anomaly = torch.randn(1, 9)
        
        # Call postprocess
        response = engine._postprocess(
            request=sample_prediction_request,
            health=health,
            degradation=degradation,
            anomaly=anomaly,
            inference_time=0.045
        )
        
        # Assertions
        assert isinstance(response, PredictionResponse)
        assert response.equipment_id == "exc_001"
        assert response.health.score == pytest.approx(0.87, abs=0.01)
        assert response.degradation.rate == pytest.approx(0.12, abs=0.01)
        assert len(response.anomaly.predictions) == 9
        assert response.inference_time_ms == pytest.approx(45.0, abs=1.0)


class TestStats:
    """Test statistics and monitoring."""
    
    @patch('src.inference.inference_engine.ModelManager')
    def test_get_stats(
        self,
        mock_manager_class,
        tmp_path
    ):
        """Test get_stats returns correct info."""
        # Setup
        ckpt_path = tmp_path / "model.ckpt"
        torch.save({}, ckpt_path)
        
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_manager.load_model = Mock(return_value=mock_model)
        mock_manager.warmup = Mock()
        mock_manager.get_model_info = Mock(return_value={
            "device": "cpu",
            "num_parameters": 2500000
        })
        mock_manager_class.return_value = mock_manager
        
        config = InferenceConfig(
            model_path=str(ckpt_path),
            device="cpu",
            batch_size=32
        )
        
        engine = InferenceEngine(config=config)
        
        # Get stats
        stats = engine.get_stats()
        
        # Assertions
        assert "model_path" in stats
        assert "device" in stats
        assert "batch_size" in stats
        assert stats["batch_size"] == 32
        assert "model_parameters" in stats
