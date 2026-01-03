"""Quick-start test for node-centric dataset pipeline.

Tests:
  ✓ Load raw UCI data
  ✓ Resample to 10 Hz
  ✓ Extract semisynthetic features
  ✓ Build PyG Data graphs
  ✓ Create DataLoader

Run:
  python -m src.v2_clean.test_quick_start
"""

import logging
from pathlib import Path

try:
    from torch_geometric.data import DataLoader
    import torch
except ImportError:
    raise ImportError("Please install torch and torch-geometric: pip install torch torch-geometric")

from .data import RawDataLoader, SENSOR_ORDER
from .dataset import NodeCentricGraphDataset
from .topology import NodeCentricTopology

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_raw_data_loading():
    """Test loading raw UCI data."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Raw Data Loading")
    logger.info("=" * 60)

    data_dir = Path("services/gnn_service/data/raw_real_dataset")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False

    try:
        loader = RawDataLoader(data_dir)
        logger.info(f"✓ Loaded {len(loader.sensor_data)} sensors")
        logger.info(f"  Sensors: {list(loader.sensor_data.keys())}")

        for sensor, data in loader.sensor_data.items():
            logger.info(f"    {sensor}: {len(data):>8} samples @ 100 Hz")

        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def test_cycle_loading():
    """Test loading cycles with resampling."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Cycle Loading & Resampling")
    logger.info("=" * 60)

    data_dir = Path("services/gnn_service/data/raw_real_dataset")
    try:
        loader = RawDataLoader(data_dir)
        cycles = loader.load_cycles()

        logger.info(f"✓ Loaded {len(cycles)} cycles")
        if cycles:
            cycle = cycles[0]
            logger.info(f"\nFirst cycle details:")
            logger.info(f"  ID: {cycle.cycle_id}")
            logger.info(f"  Data shape: {cycle.data.shape}")
            logger.info(f"  Expected: (600, 17)")
            logger.info(f"  Label: {cycle.label}")
            logger.info(f"  Metadata: {cycle.metadata}")

            # Check shapes
            assert cycle.data.shape == (600, 17), f"Expected (600, 17), got {cycle.data.shape}"
            logger.info(f"✓ Data shape correct")

            return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def test_topology():
    """Test node-centric topology."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Node-Centric Topology")
    logger.info("=" * 60)

    try:
        topology = NodeCentricTopology()
        logger.info(f"✓ Created topology: {topology}")
        logger.info(f"  Nodes: {len(topology.nodes)}")
        logger.info(f"  Edges: {len(topology.edges)}")
        logger.info(f"\n  Node list:")
        for i, node in enumerate(topology.nodes):
            logger.info(f"    [{i:2d}] {node}")

        # Check edge index
        source, target = topology.get_edge_index()
        logger.info(f"\n  Edge index shape: ({len(source)}, {len(target)})")
        logger.info(f"  First 5 edges:")
        for i in range(min(5, len(source))):
            src_name = topology.nodes[source[i]]
            tgt_name = topology.nodes[target[i]]
            logger.info(f"    {src_name} → {tgt_name}")

        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def test_dataset():
    """Test NodeCentricGraphDataset."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: NodeCentricGraphDataset")
    logger.info("=" * 60)

    data_dir = Path("services/gnn_service/data/raw_real_dataset")
    try:
        dataset = NodeCentricGraphDataset(
            raw_data_dir=str(data_dir),
            split="train",
            normalize_features=True,
        )

        logger.info(f"✓ Created dataset: {dataset.split} split")
        logger.info(f"  Total cycles: {len(dataset.cycles)}")
        logger.info(f"  Train samples: {len(dataset)}")
        logger.info(f"  Topology: {dataset.topology}")

        # Get first sample
        logger.info(f"\n  Fetching first sample...")
        sample = dataset[0]
        logger.info(f"  ✓ Got sample")
        logger.info(f"    x (node features): {sample.x.shape}")
        logger.info(f"      Expected: (17, 48)")
        logger.info(f"    edge_index: {sample.edge_index.shape}")
        logger.info(f"      Expected: (2, 22)")
        logger.info(f"    y (labels): {sample.y.shape}")
        logger.info(f"      Expected: (4,)")

        # Validate shapes
        assert sample.x.shape == (17, 48), f"Expected (17, 48), got {sample.x.shape}"
        assert sample.edge_index.shape == (2, 22), f"Expected (2, 22), got {sample.edge_index.shape}"
        assert sample.y.shape == (4,), f"Expected (4,), got {sample.y.shape}"
        logger.info(f"  ✓ All shapes correct")

        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def test_dataloader():
    """Test PyTorch DataLoader."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: PyTorch DataLoader")
    logger.info("=" * 60)

    data_dir = Path("services/gnn_service/data/raw_real_dataset")
    try:
        dataset = NodeCentricGraphDataset(
            raw_data_dir=str(data_dir),
            split="train",
            normalize_features=True,
        )

        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        logger.info(f"✓ Created DataLoader")
        logger.info(f"  Batch size: 4")
        logger.info(f"  Total batches: {len(loader)}")

        # Get first batch
        batch = next(iter(loader))
        logger.info(f"\n  First batch:")
        logger.info(f"    x: {batch.x.shape}")
        logger.info(f"    edge_index: {batch.edge_index.shape}")
        logger.info(f"    y: {batch.y.shape}")
        logger.info(f"    batch (graph index): {batch.batch.shape}")

        logger.info(f"  ✓ Batch shapes valid")

        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "#" * 60)
    logger.info("# NODE-CENTRIC HYDRAULIC DATASET - QUICK START TEST")
    logger.info("#" * 60)

    results = []
    results.append(("Raw Data Loading", test_raw_data_loading()))
    results.append(("Cycle Loading", test_cycle_loading()))
    results.append(("Topology", test_topology()))
    results.append(("Dataset", test_dataset()))
    results.append(("DataLoader", test_dataloader()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    all_passed = all(result for _, result in results)
    logger.info("\n" + ("=" * 60))
    if all_passed:
        logger.info("✓ ALL TESTS PASSED - Ready for training!")
    else:
        logger.info("✗ Some tests failed - see errors above")
    logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
