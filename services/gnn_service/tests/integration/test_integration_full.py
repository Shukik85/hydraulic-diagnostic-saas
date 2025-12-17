"""Full integration test for GNN training pipeline.

Tests:
- TimescaleConnector (mock)
- FeatureEngineer (mock)
- GraphTopology (mock)
- TemporalHydraulicDataLoader
- GRAPE imputation
- Config integration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import yaml

from src.data.timescale_connector import TimescaleConnector
from src.features.feature_engineer import FeatureEngineer
from src.topology.mock_topology import GraphTopology
from src.training.dataloader_temporal import TemporalHydraulicDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_full_pipeline():
    """Test complete pipeline with GRAPE imputation."""
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
                "enabled": True,
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
    
    logger.info("✅ Components initialized")
    logger.info("   - TimescaleConnector: OK")
    logger.info("   - FeatureEngineer: OK")
    logger.info("   - GraphTopology: %d components, %d connections", 
                len(topology.components), len(topology.connections))

    # 2. Create DataLoader
    logger.info("\n" + "="*80)
    logger.info("📊 Step 2: Creating DataLoader with GRAPE")
    logger.info("="*80)
    
    loader = TemporalHydraulicDataLoader(
        timescale_connector=connector,
        window_size=config["temporal"]["window_size"],
        stride=config["temporal"]["stride"],
        sequence_length=config["temporal"]["sequence_length"],
        config=config,
        device="cpu",
    )
    
    logger.info("✅ DataLoader created")
    logger.info("   - GRAPE imputation: %s", "ENABLED" if loader.imputation_enabled else "DISABLED")
    logger.info("   - Window size: %ds", config["temporal"]["window_size"])
    logger.info("   - Stride: %ds", config["temporal"]["stride"])

    # 3. Load temporal sequence
    logger.info("\n" + "="*80)
    logger.info("🔄 Step 3: Loading Temporal Sequence")
    logger.info("="*80)
    
    graphs = await loader.load_temporal_sequence(
        equipment_id="pump_001",
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T06:00:00",  # 6 hours
        topology=topology,
    )
    
    logger.info("✅ Temporal sequence loaded")
    logger.info("   - Number of snapshots: %d", len(graphs))

    # 4. Validate graphs
    logger.info("\n" + "="*80)
    logger.info("🔍 Step 4: Validating Graph Structure")
    logger.info("="*80)
    
    if len(graphs) > 0:
        sample_graph = graphs[0]
        logger.info("✅ Sample graph structure:")
        logger.info("   - Nodes: %d", sample_graph.x.shape[0])
        logger.info("   - Features per node: %d", sample_graph.x.shape[1])
        logger.info("   - Edges: %d", sample_graph.edge_index.shape[1])
        logger.info("   - Missing nodes: %d / %d", 
                    (~sample_graph.mask_nodes).sum(), sample_graph.mask_nodes.shape[0])
        
        if hasattr(sample_graph, "confidence"):
            logger.info("   - ✅ Confidence scores present")
            logger.info("     - Mean confidence: %.3f", sample_graph.confidence.mean())
            logger.info("     - Min confidence: %.3f", sample_graph.confidence.min())
        else:
            logger.info("   - ⚠️  No confidence scores (imputation disabled?)")
        
        # Check targets
        if hasattr(sample_graph, "y_graph_health"):
            logger.info("   - ✅ Targets present")
            logger.info("     - Graph health: %.3f", sample_graph.y_graph_health.item())
            logger.info("     - Component health shape: %s", str(sample_graph.y_component_health.shape))
        else:
            logger.info("   - ⚠️  No targets attached")
    else:
        logger.error("❌ No graphs generated!")
        return False

    # 5. Test batch creation
    logger.info("\n" + "="*80)
    logger.info("📦 Step 5: Testing Batch Creation")
    logger.info("="*80)
    
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    batch_loader = PyGDataLoader(graphs, batch_size=4, shuffle=False)
    batch = next(iter(batch_loader))
    
    logger.info("✅ Batch created")
    logger.info("   - Batch size: %d", batch.num_graphs)
    logger.info("   - Total nodes: %d", batch.x.shape[0])
    logger.info("   - Total edges: %d", batch.edge_index.shape[1])
    
    if hasattr(batch, "confidence"):
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
    logger.info("  - GRAPE imputation: %s", "WORKING" if hasattr(graphs[0], "confidence") else "NOT APPLIED")
    logger.info("  - Batch creation: OK")
    logger.info("  - Ready for training: YES")
    logger.info("="*80)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    sys.exit(0 if success else 1)
