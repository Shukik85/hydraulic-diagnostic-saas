"""
Test suite for Universal Temporal GNN model
"""
import pytest
import torch
from model_universal_temporal import UniversalTemporalGNN, create_model, get_feature_dim

def test_feature_dim_calculation():
    metadata = {
        "components": [
            {"id": "pump", "sensors": ["pressure", "flow", "temp"]},
            {"id": "valve", "sensors": ["pressure", "position"]},
        ]
    }
    dim = get_feature_dim(metadata)
    assert dim == 15  # 3 unique sensors * 5 stats

def test_model_creation():
    metadata = {
        "components": [
            {"id": "pump", "sensors": ["pressure"]},
        ]
    }
    model = create_model(metadata, device="cpu", use_compile=False)
    assert model is not None
    assert model.n_nodes == 1

def test_forward_pass():
    metadata = {
        "components": [
            {"id": "pump", "sensors": ["pressure", "flow"]},
            {"id": "valve", "sensors": ["pressure"]},
        ]
    }
    model = create_model(metadata, device="cpu", use_compile=False)
    
    x = torch.randn(1, 12, 2, 10)  # batch, time, nodes, features
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    
    health, degradation = model(x, edge_index)
    
    assert health.shape == (1, 2)
    assert degradation.shape == (1, 2)
    assert torch.all((health >= 0) & (health <= 1))
    assert torch.all((degradation >= -1) & (degradation <= 1))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
