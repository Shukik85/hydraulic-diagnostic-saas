"""
Test suite for inference service
"""
import pytest
import torch
from inference_service import InferenceEngine
from model_universal_temporal import create_model

@pytest.fixture
def metadata():
    return {
        "components": [
            {"id": "pump", "sensors": ["pressure", "flow"]},
            {"id": "valve", "sensors": ["pressure"]},
        ]
    }

@pytest.fixture
def engine(metadata):
    model = create_model(metadata, device="cpu", use_compile=False)
    return InferenceEngine(model, device="cpu")

def test_engine_initialization(engine):
    assert engine.model is not None
    assert engine.device is not None

def test_query_timescaledb(engine):
    """Test TimescaleDB query (dummy data)."""
    data = engine.query_timescaledb("test_system", window_minutes=60)
    assert isinstance(data, dict)
    assert len(data) > 0

def test_extract_temporal_features(engine, metadata):
    """Test feature extraction."""
    sensor_data = {
        "pump": [{"pressure": 250.0, "flow": 80.0} for _ in range(12)],
        "valve": [{"pressure": 200.0} for _ in range(12)],
    }
    
    x_seq, edge_index = engine.extract_temporal_features(sensor_data, metadata)
    
    assert x_seq.shape[0] == 1  # batch
    assert x_seq.shape[1] == 12  # timesteps
    assert x_seq.shape[2] == 2  # nodes
    assert edge_index.shape[0] == 2

def test_predict(engine, metadata):
    """Test inference prediction."""
    x = torch.randn(1, 12, 2, 10)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    
    health, degradation = engine.predict(x, edge_index)
    
    assert isinstance(health, dict)
    assert isinstance(degradation, dict)
    assert len(health) > 0
    assert all(0 <= v <= 1 for v in health.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
