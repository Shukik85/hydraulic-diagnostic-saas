"""Trainer factory functions for GNN training.

Provides:
- create_trainer: General trainer
- create_production_trainer: Production settings from config
- create_development_trainer: Development/debug settings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

logger = logging.getLogger(__name__)


class TrainerConfig:
    """Configuration for Trainer."""

    def __init__(
        self,
        max_epochs: int = 100,
        devices: int | list[int] = 1,
        accelerator: str = "auto",
        strategy: str = "auto",
        precision: str | int = "32",
        # NOTE: gradient_clip_val removed - done manually in lightning_module
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int = 2,
        deterministic: bool = False,
        benchmark: bool = True,
        # Callbacks
        enable_checkpointing: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_filename: str = "hydraulic-gnn-{epoch:02d}-{val_total_loss:.4f}",
        checkpoint_monitor: str = "val/total_loss",
        checkpoint_mode: str = "min",
        save_top_k: int = 5,
        save_last: bool = True,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 30,
        early_stopping_min_delta: float = 0.0001,
        # Logging
        logger_save_dir: str | Path = "logs",
        logger_name: str = "hydraulic_gnn",
        **kwargs: Any,
    ):
        self.max_epochs = max_epochs
        self.devices = devices
        self.accelerator = accelerator
        self.strategy = strategy
        self.precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.num_sanity_val_steps = num_sanity_val_steps
        self.deterministic = deterministic
        self.benchmark = benchmark

        # Callbacks
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Logging
        self.logger_save_dir = Path(logger_save_dir)
        self.logger_name = logger_name

        # Extra kwargs
        self.extra_kwargs = kwargs


def create_trainer(config: TrainerConfig) -> pl.Trainer:
    """Create PyTorch Lightning Trainer from config.

    Args:
        config: Trainer configuration

    Returns:
        Configured Trainer

    Examples:
        >>> config = TrainerConfig(max_epochs=100, devices=1)
        >>> trainer = create_trainer(config)
        >>> trainer.fit(module, train_loader, val_loader)
    """
    callbacks = []

    # Progress bar
    callbacks.append(RichProgressBar())

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Checkpoint callback
    if config.enable_checkpointing:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=config.checkpoint_filename,
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_top_k=config.save_top_k,
            save_last=config.save_last,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    if config.enable_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=config.checkpoint_monitor,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=config.checkpoint_mode,
            verbose=False,
        )
        callbacks.append(early_stop_callback)

    # Logger
    config.logger_save_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=config.logger_save_dir,
        name=config.logger_name,
    )

    # Strategy
    strategy = config.strategy
    if isinstance(config.devices, int) and config.devices > 1 and strategy == "auto":
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
        )

    # Create trainer (NO gradient_clip_val for manual optimization compatibility)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=strategy,
        precision=config.precision,
        # NOTE: gradient_clip_val removed - done manually in training_step
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        num_sanity_val_steps=config.num_sanity_val_steps,
        deterministic=config.deterministic,
        benchmark=config.benchmark,
        callbacks=callbacks,
        logger=tb_logger,
        **config.extra_kwargs,
    )

    return trainer


def create_production_trainer(
    config: dict | None = None,
) -> pl.Trainer:
    """Create production trainer from full config dictionary.

    Args:
        config: Full configuration dictionary from training_temporal.yaml

    Returns:
        Production-ready Trainer

    Examples:
        >>> import yaml
        >>> with open("configs/training_temporal.yaml") as f:
        ...     config = yaml.safe_load(f)
        >>> trainer = create_production_trainer(config=config)
    """
    if config is None:
        # Fallback to defaults
        logger.warning("No config provided, using defaults")
        config = {
            "training": {"max_epochs": 200, "devices": 1},
            "checkpoint": {"dirpath": "checkpoints/production"},
            "logging": {"save_dir": "logs/production"},
        }

    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpoint", {})
    early_stopping_cfg = config.get("early_stopping", {})
    logging_cfg = config.get("logging", {})
    validation_cfg = config.get("validation", {})
    hardware_cfg = config.get("hardware", {})  # noqa: F841
    reproducibility_cfg = config.get("reproducibility", {})

    trainer_config = TrainerConfig(
        # Training
        max_epochs=training_cfg.get("max_epochs", 200),
        devices=training_cfg.get("devices", 1),
        accelerator=training_cfg.get("accelerator", "gpu"),
        precision=training_cfg.get("precision", 16),
        # NOTE: gradient_clip_val removed
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        # Checkpoint
        enable_checkpointing=True,
        checkpoint_dir=checkpoint_cfg.get("dirpath", "checkpoints/production"),
        checkpoint_filename=checkpoint_cfg.get(
            "filename", "hydraulic-gnn-{epoch:02d}-{val_total_loss:.4f}"
        ),
        checkpoint_monitor=checkpoint_cfg.get("monitor", "val/total_loss"),
        checkpoint_mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=checkpoint_cfg.get("save_top_k", 5),
        save_last=checkpoint_cfg.get("save_last", True),
        # Early stopping
        enable_early_stopping=early_stopping_cfg.get("enabled", True),
        early_stopping_patience=early_stopping_cfg.get("patience", 30),
        early_stopping_min_delta=early_stopping_cfg.get("min_delta", 0.0001),
        # Logging
        logger_save_dir=logging_cfg.get("save_dir", "logs/production"),
        logger_name=logging_cfg.get("name", "hydraulic_gnn_production"),
        log_every_n_steps=logging_cfg.get("log_every_n_steps", 50),
        # Validation
        check_val_every_n_epoch=int(validation_cfg.get("check_interval", 1.0)),
        num_sanity_val_steps=validation_cfg.get("num_sanity_val_steps", 2),
        # Reproducibility
        deterministic=reproducibility_cfg.get("deterministic", False),
        benchmark=reproducibility_cfg.get("benchmark", True),
    )

    return create_trainer(trainer_config)


def create_development_trainer(
    config: dict | None = None,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """Create development trainer for debugging.

    Args:
        config: Full configuration dictionary
        fast_dev_run: Run single batch for debugging

    Returns:
        Development Trainer
        
    NOTE:
        - num_sanity_val_steps=0 disables sanity checks
        - This prevents validation BEFORE training starts
        - Validation before training can cause double backward errors
        - For full debugging, use production trainer instead
    """
    if config is None:
        config = {"training": {"max_epochs": 10, "devices": 1}}

    training_cfg = config.get("training", {})

    trainer_config = TrainerConfig(
        max_epochs=training_cfg.get("max_epochs", 10),
        devices=training_cfg.get("devices", 1),
        accelerator="cpu",  # ALWAYS use CPU for dev mode
        precision="32",
        enable_checkpointing=False,
        enable_early_stopping=False,
        logger_save_dir="logs/dev",
        logger_name="hydraulic_gnn_dev",
        num_sanity_val_steps=0,  # 🔥 CRITICAL: Disable sanity checks to prevent double backward
        fast_dev_run=fast_dev_run,
    )

    return create_trainer(trainer_config)
