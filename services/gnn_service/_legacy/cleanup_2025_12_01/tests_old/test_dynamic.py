# services/gnn_service/tests/test_dynamic.py
"""
Integration test for dynamic GNN pipeline
"""
import torch
from model_dynamic_gnn import create_model
from schemas import EquipmentMetadata

def test_dynamic_model_forward():
    # Minimal 2-component metadata
    metadata = EquipmentMetadata(
        equipment_id="demo",
        equipment_type="simple",
        manufacturer="test",
        model="A1",
        components=[
            dict(id="pump", type="pump", role="energy_source", sensors=[], parameters={}, criticality=1.0),
            dict(id="cylinder", type="cylinder", role="actuator", sensors=[], parameters={}, criticality=1.0)
        ],
        connections=[],
        energy_sources=[dict(component_id="pump", operation_modes=[], priority=1, activation_strategy="manual")],
    )
    model = create_model(metadata, device="cpu")
    comp_feats = {"pump": torch.zeros(1,5,3), "cylinder": torch.zeros(1,5,3)}
    health, deg, attn = model(comp_feats)
    assert health.shape[-1] == 2
    assert deg.shape[-1] == 2
