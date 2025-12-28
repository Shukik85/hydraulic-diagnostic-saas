"""Full integration test for GNN training pipeline.

Tests:
- TimescaleConnector (mock)
- FeatureEngineer (mock)
- GraphTopology (mock)
- TemporalHydraulicDataLoader
- GRAPE imputation (optional)
- Config integration
- Graph structure validation
- Batch creation
"""

import logging
from pathlib import Path

import pytest
import torch
import yaml

from src.data.timescale_connector import TimescaleConnector
from src.features.feature_engineer import FeatureEngineer
from src.topology.mock_topology import GraphTopology
from src.training.dataloader_temporal import TemporalHydraulicDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline():
    """Test complete pipeline with GRAPE imputation.
    
    Validates:
    - Component initialization
    - Temporal graph generation
    - Graph structure (nodes, edges, features, masks)
    - GRAPE imputation (if enabled)
    - Batch creation
    - Target attachment
    - No NaN values
    """
    logger.info("="*80)
    logger.info("🚀 Starting Full Integration Test")
    logger.info("="*80)

    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "training_temporal.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded config from %s", config_path)
    else:
        logger.warning("⚠️  Config not found, using minimal config")
        config = {
            "imputation": {
                "enabled": False,  # Disable for faster test
                "spatial": {"hidden_dim": 128, "use_static_prior": True},
                "temporal": {"hidden_dim": 128},
            },
            "temporal": {
                "window_size": 3600,
                "stride": 900,
                "sequence_length": 12,
            },
        }

    # 1. Initialize components
    logger.info("\n" + "="*80)
    logger.info("📦 Step 1: Initializing Components")
    logger.info("="*80)
    
    connector = TimescaleConnector(seed=42)
    engineer = FeatureEngineer()
    topology = GraphTopology.create_mock_excavator(equipment_id="pump_001")
    
    # Assertions: components created successfully
    assert connector is not None, "TimescaleConnector failed to initialize"
    assert engineer is not None, "FeatureEngineer failed to initialize"
    assert topology is not None, "GraphTopology failed to initialize"
    assert len(topology.components) == 10, f"Expected 10 components, got {len(topology.components)}"
    assert len(topology.connections) == 12, f"Expected 12 connections, got {len(topology.connections)}"
    
    logger.info("✅ Components initialized")
    logger.info("   - TimescaleConnector: OK")
    logger.info("   - FeatureEngineer: OK")
    logger.info("   - GraphTopology: %d components, %d connections", 
                len(topology.components), len(topology.connections))

    # 2. Create DataLoader
    logger.info("\n" + "="*80)
    logger.info("📊 Step 2: Creating DataLoader")
    logger.info("="*80)
    
    loader = TemporalHydraulicDataLoader(
        timescale_connector=connector,
        window_size=config["temporal"]["window_size"],
        stride=config["temporal"]["stride"],
        sequence_length=config["temporal"]["sequence_length"],
        config=config,
        device="cpu",
    )
    
    assert loader is not None, "DataLoader failed to initialize"
    
    logger.info("✅ DataLoader created")
    logger.info("   - Imputation: %s", "ENABLED" if loader.imputation_enabled else "DISABLED")
    logger.info("   - Window size: %ds", config["temporal"]["window_size"])
    logger.info("   - Stride: %ds", config["temporal"]["stride"])

    # 3. Load temporal sequence
    logger.info("\n" + "="*80)
    logger.info("🔄 Step 3: Loading Temporal Sequence")
    logger.info("="*80)
    
    graphs = await loader.load_temporal_sequence(
        equipment_id="pump_001",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-01T06:00:00+00:00",  # 6 hours
        topology=topology,
    )
    
    # Assertions: graphs generated
    assert len(graphs) > 0, "No graphs generated - data loading failed"
    
    logger.info("✅ Temporal sequence loaded")
    logger.info("   - Number of snapshots: %d", len(graphs))

    # 4. Validate graph structure
    logger.info("\n" + "="*80)
    logger.info("🔍 Step 4: Validating Graph Structure")
    logger.info("="*80)
    
    sample_graph = graphs[0]
    
    # Assertions: graph structure
    assert hasattr(sample_graph, 'x'), "Graph missing node features 'x'"
    assert hasattr(sample_graph, 'edge_index'), "Graph missing 'edge_index'"
    assert hasattr(sample_graph, 'mask_nodes'), "Graph missing 'mask_nodes'"
    
    assert sample_graph.x.shape[0] > 0, "Graph has 0 nodes"
    assert sample_graph.x.shape[1] == 34, f"Expected 34 features per node, got {sample_graph.x.shape[1]}"
    assert sample_graph.edge_index.shape[0] == 2, f"edge_index should be [2, E], got {sample_graph.edge_index.shape}"
    assert sample_graph.edge_index.shape[1] > 0, "Graph has 0 edges"
    assert sample_graph.mask_nodes.shape[0] == sample_graph.x.shape[0], "mask_nodes length mismatch"
    
    # Validate no NaN values
    assert not torch.isnan(sample_graph.x).any(), "Node features contain NaN values"
    if hasattr(sample_graph, 'edge_attr'):
        assert not torch.isnan(sample_graph.edge_attr).any(), "Edge features contain NaN values"
    
    logger.info("✅ Sample graph structure:")
    logger.info("   - Nodes: %d", sample_graph.x.shape[0])
    logger.info("   - Features per node: %d", sample_graph.x.shape[1])
    logger.info("   - Edges: %d", sample_graph.edge_index.shape[1])
    logger.info("   - Missing nodes: %d / %d", 
                (~sample_graph.mask_nodes).sum(), sample_graph.mask_nodes.shape[0])
    
    # Check confidence scores (if imputation enabled)
    if hasattr(sample_graph, "confidence"):
        assert sample_graph.confidence.shape[0] == sample_graph.x.shape[0], "Confidence length mismatch"
        assert (sample_graph.confidence >= 0).all() and (sample_graph.confidence <= 1).all(), \
            "Confidence scores out of range [0, 1]"
        logger.info("   - ✅ Confidence scores present")
        logger.info("     - Mean confidence: %.3f", sample_graph.confidence.mean())
        logger.info("     - Min confidence: %.3f", sample_graph.confidence.min())
    else:
        logger.info("   - ⚠️  No confidence scores (imputation disabled)")
    
    # Check targets
    if hasattr(sample_graph, "y_graph_health"):
        assert sample_graph.y_graph_health.shape[0] == 1, "Graph health target shape mismatch"
        assert hasattr(sample_graph, "y_component_health"), "Missing component health targets"
        assert sample_graph.y_component_health.shape[0] == sample_graph.x.shape[0], \
            "Component health target length mismatch"
        logger.info("   - ✅ Targets present")
        logger.info("     - Graph health: %.3f", sample_graph.y_graph_health.item())
        logger.info("     - Component health shape: %s", str(sample_graph.y_component_health.shape))
    else:
        logger.info("   - ⚠️  No targets attached")

    # 5. Test batch creation
    logger.info("\n" + "="*80)
    logger.info("📦 Step 5: Testing Batch Creation")
    logger.info("="*80)
    
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    batch_loader = PyGDataLoader(graphs, batch_size=min(4, len(graphs)), shuffle=False)
    batch = next(iter(batch_loader))
    
    # Assertions: batch structure
    assert batch.num_graphs > 0, "Batch has 0 graphs"
    assert batch.x.shape[0] > 0, "Batch has 0 nodes"
    assert batch.edge_index.shape[1] > 0, "Batch has 0 edges"
    assert not torch.isnan(batch.x).any(), "Batch node features contain NaN"
    
    logger.info("✅ Batch created")
    logger.info("   - Batch size: %d", batch.num_graphs)
    logger.info("   - Total nodes: %d", batch.x.shape[0])
    logger.info("   - Total edges: %d", batch.edge_index.shape[1])
    
    if hasattr(batch, "confidence"):
        assert batch.confidence.shape[0] == batch.x.shape[0], "Batch confidence length mismatch"
        logger.info("   - ✅ Confidence in batch: %s", str(batch.confidence.shape))
    else:
        logger.info("   - ⚠️  No confidence in batch")

    # 6. Summary
    logger.info("\n" + "="*80)
    logger.info("✅ Integration Test PASSED")
    logger.info("="*80)
    logger.info("Summary:")
    logger.info("  - Mock components: OK")
    logger.info("  - Temporal snapshots: %d", len(graphs))
    logger.info("  - Imputation: %s", "WORKING" if hasattr(graphs[0], "confidence") else "DISABLED")
    logger.info("  - Batch creation: OK")
    logger.info("  - No NaN values: OK")
    logger.info("  - Ready for training: YES")
    logger.info("="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_full_pipeline())
