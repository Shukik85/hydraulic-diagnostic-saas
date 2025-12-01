import torch
import pandas as pd
from dataset_dynamic import DynamicTemporalGraphDataset
from schemas import EquipmentMetadata

def test_dataset_dynamic():
    # Create dummy CSV
    import tempfile
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="5T"),
        "pump_pressure": [25]*10,
        "cylinder_main_pressure": [12]*10
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    # Simple metadata
    meta = EquipmentMetadata(
        equipment_id="simple-test",
        equipment_type="lift",
        manufacturer="test",
        model="t",
        components=[
            dict(id="pump", type="pump", role="energy_source", sensors=[{"type": "pressure", "location": "outlet", "operating_range": {"min": 15, "max": 35, "nominal": 25, "unit": "bar"}}], parameters={}, criticality=1.0),
            dict(id="cylinder_main", type="cylinder", role="actuator", sensors=[{"type": "pressure", "location": "piston_side", "operating_range": {"min": 5, "max": 20, "nominal": 12, "unit": "bar"}}], parameters={}, criticality=1.0)
        ],
        connections=[],
        energy_sources=[dict(component_id="pump", operation_modes=[], priority=1, activation_strategy="manual")],
    )
    dataset = DynamicTemporalGraphDataset(tmp_path, meta, sequence_length=5)
    assert len(dataset) > 0
