"""Phase 2 compatibility check script.

Verifies that ModelConfig + UniversalTemporalGNNv2 + HydraulicGNNModule
are compatible after Phase 2 migration.

Usage:
    python scripts/check_phase2_compatibility.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch_geometric.data import Data

try:
    from models import ModelConfig, UniversalTemporalGNNv2
    from training.lightning_module import HydraulicGNNModule
    
    print("✅ Imports successful")
    print(f"   - ModelConfig")
    print(f"   - UniversalTemporalGNNv2")
    print(f"   - HydraulicGNNModule")
    print()
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 1: ModelConfig creation
print("Test 1: ModelConfig creation")
try:
    config = ModelConfig(
        node_features=34,
        edge_features=14,
        gat_hidden_dim=128,
        lstm_hidden_dim=256,
        graph_anomaly_classes=9,
        component_anomaly_classes=9,
    )
    print(f"   ✅ ModelConfig created (v{config.version})")
    print(f"      - Graph anomaly classes: {config.graph_anomaly_classes}")
    print(f"      - Component anomaly classes: {config.component_anomaly_classes}")
    print()
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: UniversalTemporalGNNv2 creation
print("Test 2: UniversalTemporalGNNv2 creation")
try:
    model = UniversalTemporalGNNv2(config)
    print(f"   ✅ Model created")
    print(f"      - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: HydraulicGNNModule creation
print("Test 3: HydraulicGNNModule creation")
try:
    lightning_module = HydraulicGNNModule(
        model_config=config,
        use_advanced_losses=True,
        use_confidence_weighting=False,  # Disable for simplicity
    )
    print(f"   ✅ Lightning module created")
    print(f"      - Parameters: {sum(p.numel() for p in lightning_module.parameters()):,}")
    print()
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Forward pass
print("Test 4: Forward pass (single graph)")
try:
    # Create dummy data
    num_nodes = 10
    num_edges = 15
    
    data = Data(
        x=torch.randn(num_nodes, 34),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 14),
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(data, temporal=False)
    
    print(f"   ✅ Forward pass successful")
    print(f"      - Output keys: {list(outputs.keys())}")
    print()
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Output structure validation
print("Test 5: Output structure validation")
try:
    # Check nested structure
    assert 'component' in outputs, "Missing 'component' key"
    assert 'graph' in outputs, "Missing 'graph' key"
    
    # Check component keys
    assert 'health' in outputs['component'], "Missing 'component.health'"
    assert 'anomaly' in outputs['component'], "Missing 'component.anomaly'"
    
    # Check graph keys
    assert 'health' in outputs['graph'], "Missing 'graph.health'"
    assert 'degradation' in outputs['graph'], "Missing 'graph.degradation'"
    assert 'anomaly' in outputs['graph'], "Missing 'graph.anomaly'"
    assert 'rul' in outputs['graph'], "Missing 'graph.rul'"
    
    # Check shapes
    component_health = outputs['component']['health']
    component_anomaly = outputs['component']['anomaly']
    graph_health = outputs['graph']['health']
    graph_degradation = outputs['graph']['degradation']
    graph_anomaly = outputs['graph']['anomaly']
    graph_rul = outputs['graph']['rul']
    
    assert component_health.shape == (num_nodes, 1), f"Wrong component health shape: {component_health.shape}"
    assert component_anomaly.shape == (num_nodes, 9), f"Wrong component anomaly shape: {component_anomaly.shape}"
    assert graph_health.shape == (1, 1), f"Wrong graph health shape: {graph_health.shape}"
    assert graph_degradation.shape == (1, 1), f"Wrong graph degradation shape: {graph_degradation.shape}"
    assert graph_anomaly.shape == (1, 9), f"Wrong graph anomaly shape: {graph_anomaly.shape}"
    assert graph_rul.shape == (1, 1), f"Wrong graph RUL shape: {graph_rul.shape}"
    
    print(f"   ✅ Output structure valid")
    print(f"      Component-level:")
    print(f"        - health: {component_health.shape}")
    print(f"        - anomaly: {component_anomaly.shape}")
    print(f"      Graph-level:")
    print(f"        - health: {graph_health.shape}")
    print(f"        - degradation: {graph_degradation.shape}")
    print(f"        - anomaly: {graph_anomaly.shape}")
    print(f"        - RUL: {graph_rul.shape}")
    print()
except AssertionError as e:
    print(f"   ❌ Validation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Temporal mode
print("Test 6: Temporal mode forward pass")
try:
    # Create sequence of 3 timesteps
    sequence = [
        Data(
            x=torch.randn(num_nodes, 34),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 14),
            batch=torch.zeros(num_nodes, dtype=torch.long),
        )
        for _ in range(3)
    ]
    
    with torch.no_grad():
        outputs_temporal = model(sequence, temporal=True)
    
    # Validate structure
    assert 'component' in outputs_temporal
    assert 'graph' in outputs_temporal
    assert outputs_temporal['component']['health'].shape == (num_nodes, 1)
    assert outputs_temporal['graph']['rul'].shape == (1, 1)
    
    print(f"   ✅ Temporal mode successful")
    print(f"      - Sequence length: 3")
    print(f"      - Output structure: Same as single mode")
    print()
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ ALL TESTS PASSED")
print("\nPhase 2 architecture is compatible!")
print("Next steps:")
print("  1. Update tests in tests/test_universal_temporal_gnn.py")
print("  2. Update MultiTaskLoss for 6 tasks (optional)")
print("  3. Update integration tests")
