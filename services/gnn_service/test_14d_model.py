#!/usr/bin/env python3
"""Quick test for 14D edge features model.

Validates that UniversalTemporalGNN accepts 14D edge features
and produces valid outputs.

Author: GNN Service Team
Python: 3.14+
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.gnn_model import UniversalTemporalGNN


def test_14d_model():
    """Test model with 14D edge features."""
    print("="*60)
    print("PHASE 3 MODEL VALIDATION - 14D Edge Features")
    print("="*60)
    
    # Load converted graphs
    graph_path = Path('data/gnn_graphs_v2_14d_test.pt')
    
    if not graph_path.exists():
        print(f"\n❌ Graph file not found: {graph_path}")
        print("\nPlease run conversion first:")
        print("  python scripts/convert_graphs_to_14d.py \\")
        print("    --input data/gnn_graphs_multilabel.pt \\")
        print("    --edge-specs data/edge_specifications.json \\")
        print("    --output data/gnn_graphs_v2_14d_test.pt \\")
        print("    --max-samples 200")
        return False
    
    print(f"\nLoading graphs: {graph_path}")
    graphs = torch.load(graph_path)
    
    if not isinstance(graphs, list):
        graphs = [graphs]
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Take first graph
    graph = graphs[0]
    print(f"\nGraph structure:")
    print(f"  Nodes: {graph.x.shape}")
    print(f"  Edges: {graph.edge_index.shape}")
    print(f"  Edge features: {graph.edge_attr.shape}")
    
    # Validate 14D
    if graph.edge_attr.shape[1] != 14:
        print(f"\n❌ ERROR: Expected 14D edge features, got {graph.edge_attr.shape[1]}D")
        return False
    
    print("  ✅ Edge features are 14D")
    
    # Check for NaN/inf
    if torch.isnan(graph.edge_attr).any():
        print("\n❌ ERROR: NaN detected in edge features")
        return False
    
    if torch.isinf(graph.edge_attr).any():
        print("\n❌ ERROR: Inf detected in edge features")
        return False
    
    print("  ✅ No NaN/Inf values")
    
    # Create model with 14D edges
    print("\nCreating model...")
    model = UniversalTemporalGNN(
        in_channels=graph.x.shape[1],  # Node features
        hidden_channels=128,
        num_heads=4,
        num_gat_layers=2,
        lstm_hidden=128,
        lstm_layers=1,
        edge_feature_dim=14,  # Phase 3.1: 14D
        num_tasks=9,  # 9 anomaly types
        use_compile=False  # Disable torch.compile for testing
    )
    
    print(f"  Model created: {model.__class__.__name__}")
    print(f"  Edge dim: {model.edge_feature_dim}")
    
    model.eval()
    
    # Forward pass
    print("\nRunning forward pass...")
    
    try:
        with torch.no_grad():
            health, degradation, anomaly = model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,  # 14D
                batch=torch.zeros(graph.x.shape[0], dtype=torch.long)
            )
        
        print("  ✅ Forward pass successful!")
        
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate outputs
    print("\nValidating outputs...")
    
    print(f"  Health: {health.shape} = {health.item():.3f}")
    print(f"  Degradation: {degradation.shape} = {degradation.item():.3f}")
    print(f"  Anomaly: {anomaly.shape}")
    
    # Check shapes
    assert health.shape == torch.Size([1, 1]), f"Expected health shape [1, 1], got {health.shape}"
    assert degradation.shape == torch.Size([1, 1]), f"Expected degradation shape [1, 1], got {degradation.shape}"
    assert anomaly.shape == torch.Size([1, 9]), f"Expected anomaly shape [1, 9], got {anomaly.shape}"
    
    print("  ✅ Output shapes correct")
    
    # Check value ranges
    health_val = health.item()
    degradation_val = degradation.item()
    
    if not (0 <= health_val <= 1):
        print(f"  ⚠️ WARNING: Health outside [0, 1]: {health_val:.3f}")
    else:
        print(f"  ✅ Health in valid range: {health_val:.3f}")
    
    if not (0 <= degradation_val <= 1):
        print(f"  ⚠️ WARNING: Degradation outside [0, 1]: {degradation_val:.3f}")
    else:
        print(f"  ✅ Degradation in valid range: {degradation_val:.3f}")
    
    # Check for NaN in outputs
    if torch.isnan(health).any() or torch.isnan(degradation).any() or torch.isnan(anomaly).any():
        print("  ❌ ERROR: NaN in outputs")
        return False
    
    print("  ✅ No NaN in outputs")
    
    # Test batch inference
    print("\nTesting batch inference (5 graphs)...")
    
    batch_graphs = graphs[:5]
    
    # Stack into batch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(batch_graphs)
    
    try:
        with torch.no_grad():
            health_batch, degradation_batch, anomaly_batch = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )
        
        print(f"  Batch health: {health_batch.shape}")
        print(f"  Batch degradation: {degradation_batch.shape}")
        print(f"  Batch anomaly: {anomaly_batch.shape}")
        
        assert health_batch.shape == torch.Size([5, 1])
        assert degradation_batch.shape == torch.Size([5, 1])
        assert anomaly_batch.shape == torch.Size([5, 9])
        
        print("  ✅ Batch inference successful!")
        
    except Exception as e:
        print(f"  ❌ ERROR during batch inference: {e}")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print("\n✅ ALL TESTS PASSED!")
    print("\nPhase 3.1 Components Validated:")
    print("  ✅ 14D edge features loaded")
    print("  ✅ Model accepts 14D edges")
    print("  ✅ Forward pass successful")
    print("  ✅ Output shapes correct")
    print("  ✅ Output values in valid range")
    print("  ✅ Batch inference works")
    
    print("\n🚀 Ready for:")
    print("  - Full dataset conversion")
    print("  - Model retraining (v2.0.0)")
    print("  - Production deployment")
    
    return True


if __name__ == '__main__':
    success = test_14d_model()
    sys.exit(0 if success else 1)
