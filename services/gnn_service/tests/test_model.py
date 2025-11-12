import torch
from model_dynamic_gnn import create_model
from schemas import EquipmentMetadata

def test_model_factory():
    metadata = EquipmentMetadata(
        equipment_id="simple-test",
        equipment_type="demo",
        manufacturer="demo",
        model="demo",
        components=[
            dict(id="pump", type="pump", role="energy_source", sensors=[], parameters={}, criticality=1.0),
            dict(id="cylinder", type="cylinder", role="actuator", sensors=[], parameters={}, criticality=1.0)
        ],
        connections=[],
        energy_sources=[dict(component_id="pump", operation_modes=[], priority=1, activation_strategy="manual")],
    )
    model = create_model(metadata)
    feats = {"pump": torch.zeros(1,5,3), "cylinder": torch.zeros(1,5,3)}
    out = model(feats)
    assert out[0].shape[-1] == 2
