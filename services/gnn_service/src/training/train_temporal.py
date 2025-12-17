"""Training script for Universal Temporal GNN (Production-ready).

Fully integrated with training_temporal.yaml configuration.
Supports:
- GRAPE two-stage imputation
- Advanced losses (AsymmetricL1, QuantileRUL, PhysicsAwareFocal)
- Domain adversarial learning
- Complete config-driven training
- Local training with mock components

Usage:
    # Development mode (mock data, fast)
    python src/training/train_temporal.py \
        --config configs/training_temporal.yaml \
        --mode dev

    # Production mode (full training)
    python src/training/train_temporal.py \
        --config configs/training_temporal.yaml \
        --mode prod

    # Quick test (1 batch)
    python src/training/train_temporal.py \
        --config configs/training_temporal.yaml \
        --mode dev \
        --fast-dev-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import torch
import yaml

from src.training.lightning_module import HydraulicGNNModule
from src.training.trainer import create_development_trainer, create_production_trainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """Validate configuration structure.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ["model", "training", "loss_config"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    required_model_fields = ["in_channels", "hidden_channels"]
    for field in required_model_fields:
        if field not in config["model"]:
            raise ValueError(f"Missing required model field: {field}")

    # Validate training config
    required_training_fields = ["max_epochs", "learning_rate"]
    for field in required_training_fields:
        if field not in config["training"]:
            raise ValueError(f"Missing required training field: {field}")

    logger.info("✅ Configuration validated successfully")


def create_module_from_config(config: dict) -> HydraulicGNNModule:
    """Create HydraulicGNNModule from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Initialized HydraulicGNNModule
    """
    model_config = config["model"]
    training_config = config["training"]
    loss_config = config["loss_config"]
    scheduler_config = config.get("scheduler", {})

    # Convert component and severity weights to tensors
    component_weights = None
    severity_weights = None

    if "physics_aware" in loss_config:
        if "component_weights" in loss_config["physics_aware"]:
            component_weights = torch.tensor(
                loss_config["physics_aware"]["component_weights"],
                dtype=torch.float32,
            )
        if "severity_weights" in loss_config["physics_aware"]:
            severity_weights = torch.tensor(
                loss_config["physics_aware"]["severity_weights"],
                dtype=torch.float32,
            )

    # Determine scheduler type
    scheduler_type = scheduler_config.get("type", "plateau")

    # Create module with all config parameters
    module = HydraulicGNNModule(
        # Model architecture
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        num_heads=model_config.get("num_heads", 8),
        num_gat_layers=model_config.get("num_gat_layers", 3),
        lstm_hidden=model_config.get("lstm_hidden", 256),
        lstm_layers=model_config.get("lstm_layers", 2),
        dropout=model_config.get("dropout", 0.1),
        # Training parameters
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 1e-5),
        scheduler_type=scheduler_type,
        # Loss configuration
        loss_weighting=loss_config.get("weighting", "uncertainty"),
        loss_weights=loss_config.get("loss_weights"),
        # Advanced loss options
        use_advanced_losses=loss_config.get("use_advanced_losses", True),
        use_confidence_weighting=loss_config.get("use_confidence_weighting", True),
        use_domain_adversarial=loss_config.get("use_domain_adversarial", False),
        # RUL loss config
        rul_tau=loss_config.get("rul", {}).get("tau", 0.7),
        # Physics-aware config
        component_weights=component_weights,
        severity_weights=severity_weights,
        # Domain adversarial config
        lambda_domain=loss_config.get("domain_adversarial", {}).get("lambda_domain", 0.5),
        num_domains=loss_config.get("domain_adversarial", {}).get("num_domains", 2),
    )

    logger.info("✅ Module created with full config integration")
    logger.info(f"   - Advanced losses: {loss_config.get('use_advanced_losses', True)}")
    logger.info(f"   - Confidence weighting: {loss_config.get('use_confidence_weighting', True)}")
    logger.info(f"   - Loss weighting: {loss_config.get('weighting', 'uncertainty')}")
    logger.info(f"   - Scheduler: {scheduler_type}")

    return module


async def create_dataloader_from_config(config: dict, mode: str = "dev"):
    """Create DataLoader from configuration with mock components.

    Args:
        config: Full configuration dictionary
        mode: Training mode ('dev' or 'prod')

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from src.data.timescale_connector import TimescaleConnector
    from src.topology.mock_topology import GraphTopology
    from src.training.dataloader_temporal import TemporalHydraulicDataLoader

    temporal_config = config.get("temporal", {})
    imputation_config = config.get("imputation", {})
    training_config = config["training"]

    logger.info("📊 Setting up DataLoader with mock components...")
    logger.info(f"   - Window size: {temporal_config.get('window_size', 3600)}s")
    logger.info(f"   - Stride: {temporal_config.get('stride', 900)}s")
    logger.info(f"   - Sequence length: {temporal_config.get('sequence_length', 12)}")
    logger.info(f"   - Batch size: {training_config.get('batch_size', 32)}")
    logger.info(f"   - Imputation enabled: {imputation_config.get('enabled', False)}")

    try:
        # Initialize mock components
        connector = TimescaleConnector(seed=42)
        topology = GraphTopology.create_mock_excavator("pump_001")

        # Create temporal dataloader
        loader = TemporalHydraulicDataLoader(
            timescale_connector=connector,
            window_size=temporal_config.get("window_size", 3600),
            stride=temporal_config.get("stride", 900),
            sequence_length=temporal_config.get("sequence_length", 12),
            correlation_threshold=temporal_config.get("correlation_threshold", 0.5),
            k_neighbors=temporal_config.get("k_neighbors", 5),
            max_missing_ratio=temporal_config.get("max_missing_ratio", 0.5),
            config=config,
            device="cpu",
        )

        # Load temporal sequence
        # Dev mode: 12 hours, Prod mode: 7 days
        hours = 12 if mode == "dev" else 168
        logger.info(f"🔄 Loading {hours} hours of synthetic data...")

        graphs = await loader.load_temporal_sequence(
            equipment_id="pump_001",
            start_time="2024-01-01T00:00:00",
            end_time=f"2024-01-01T{hours:02d}:00:00" if hours < 24 else "2024-01-08T00:00:00",
            topology=topology,
        )

        if len(graphs) == 0:
            logger.error("❌ No graphs generated!")
            return None, None

        logger.info(f"✅ Generated {len(graphs)} temporal snapshots")

        # Split train/val (80/20)
        split_idx = int(0.8 * len(graphs))
        train_graphs = graphs[:split_idx]
        val_graphs = graphs[split_idx:]

        logger.info(f"   - Train: {len(train_graphs)} snapshots")
        logger.info(f"   - Val: {len(val_graphs)} snapshots")

        # Create PyTorch Geometric DataLoaders
        batch_size = training_config.get("batch_size", 32)
        num_workers = 0  # Use 0 for Windows compatibility

        train_loader = PyGDataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = PyGDataLoader(
            val_graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        logger.info("✅ DataLoaders created successfully")
        logger.info(f"   - Train batches: {len(train_loader)}")
        logger.info(f"   - Val batches: {len(val_loader)}")

        # Validate first batch
        sample_batch = next(iter(train_loader))
        logger.info(f"   - Sample batch: {sample_batch.num_graphs} graphs, {sample_batch.x.shape[0]} nodes")
        if hasattr(sample_batch, "confidence"):
            logger.info(f"   - ✅ Confidence scores present (mean: {sample_batch.confidence.mean():.3f})")
        else:
            logger.warning("   - ⚠️  No confidence scores in batch")

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"❌ Error creating DataLoader: {e}")
        logger.exception(e)
        return None, None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Universal Temporal GNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_temporal.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "prod"],
        default="dev",
        help="Training mode (dev=debug, prod=production)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run single batch for debugging",
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🚀 Universal Temporal GNN Training")
    logger.info("="*80)

    # Load and validate config
    logger.info(f"📄 Loading config from: {args.config}")
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Create module
    logger.info("\n🔧 Creating HydraulicGNNModule...")
    try:
        module = create_module_from_config(config)
    except Exception as e:
        logger.error(f"❌ Failed to create module: {e}")
        sys.exit(1)

    # Create DataLoader
    logger.info("\n📊 Setting up DataLoader...")
    try:
        train_loader, val_loader = asyncio.run(
            create_dataloader_from_config(config, mode=args.mode)
        )
    except Exception as e:
        logger.error(f"❌ Failed to create DataLoader: {e}")
        sys.exit(1)

    if train_loader is None or val_loader is None:
        logger.error("❌ DataLoader creation failed - cannot continue")
        sys.exit(1)

    # Create trainer
    logger.info("\n⚡ Creating Trainer...")
    try:
        if args.mode == "prod":
            trainer = create_production_trainer(config=config)
            logger.info("   Mode: PRODUCTION")
        else:
            trainer = create_development_trainer(
                config=config,
                fast_dev_run=args.fast_dev_run,
            )
            logger.info("   Mode: DEVELOPMENT")
            if args.fast_dev_run:
                logger.info("   ⚡ Fast dev run: 1 batch only")
    except Exception as e:
        logger.error(f"❌ Failed to create trainer: {e}")
        sys.exit(1)

    # Log trainer configuration
    logger.info(f"   - Max epochs: {config['training']['max_epochs']}")
    logger.info(f"   - Devices: {config['training'].get('devices', 1)}")
    logger.info(f"   - Accelerator: {config['training'].get('accelerator', 'gpu')}")
    logger.info(f"   - Precision: {config['training'].get('precision', 16)}")
    logger.info(f"   - Gradient clip: {config['training'].get('gradient_clip_val', 1.0)}")

    # Start training
    logger.info("\n" + "="*80)
    logger.info("🚀 Starting training...")
    logger.info("="*80)

    try:
        trainer.fit(module, train_loader, val_loader)
        logger.info("\n" + "="*80)
        logger.info("✅ Training completed successfully!")
        logger.info("="*80)

        # Log results
        if hasattr(trainer, "checkpoint_callback"):
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path:
                logger.info(f"🏆 Best model saved: {best_model_path}")

        logger.info("\n📈 View training logs:")
        logger.info(f"   tensorboard --logdir {config.get('logging', {}).get('save_dir', 'logs')}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
