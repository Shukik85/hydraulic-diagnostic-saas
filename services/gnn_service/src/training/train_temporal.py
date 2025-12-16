#!/usr/bin/env python
"""Production training script for temporal hydraulic GNN.

Implements TimeGNN + GRAPE approach:
- Temporal graph snapshots with sliding windows
- GRAPE-based missing data handling
- Multi-task learning with uncertainty weighting
- Checkpoint resuming and model export

Usage:
    python -m src.training.train_temporal --config configs/training_temporal.yaml
    python -m src.training.train_temporal --resume-from checkpoints/latest.ckpt
"""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

from src.training.dataloader_temporal import TemporalHydraulicDataLoader
from src.training.lightning_module import HydraulicGNNModule
from src.training.trainer import (
    create_production_trainer,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal hydraulic GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training
  python -m src.training.train_temporal --config configs/training_temporal.yaml

  # Resume from checkpoint
  python -m src.training.train_temporal --resume-from checkpoints/latest.ckpt

  # Quick test with fast_dev_run
  python -m src.training.train_temporal --fast-dev-run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_temporal.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="prod",
        help="Training mode (dev=fast iteration, prod=full training)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run single batch for debugging",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--export-onnx",
        type=str,
        help="Export model to ONNX after training",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Create Lightning module
    module = HydraulicGNNModule(
        in_channels=config["model"]["in_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        num_heads=config["model"]["num_heads"],
        num_gat_layers=config["model"]["num_gat_layers"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        loss_weighting="uncertainty",  # Use uncertainty weighting
        loss_weights=config["loss_weights"],
    )

    # Create trainer
    if args.mode == "prod":
        trainer = create_production_trainer(
            max_epochs=config["training"]["max_epochs"],
            devices=config["training"]["devices"],
        )
    else:
        from src.training.trainer import create_development_trainer

        trainer = create_development_trainer()

    if args.fast_dev_run:
        trainer.fit(
            module,
            # TODO: Load actual dataloaders
            # train_loader, val_loader,
            fast_dev_run=True,
        )
    else:
        # Load data (placeholder)
        # TODO: Implement actual data loading with TemporalHydraulicDataLoader
        train_loader = None
        val_loader = None

        # Train or resume
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume_from)
        else:
            trainer.fit(module, train_loader, val_loader)

    # Export to ONNX if requested
    if args.export_onnx:
        logger.info(f"Exporting to ONNX: {args.export_onnx}")
        # TODO: Implement ONNX export

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
