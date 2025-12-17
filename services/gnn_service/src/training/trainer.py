"""Trainer factory functions for GNN training.

Provides:
- create_trainer: General trainer
- create_production_trainer: Production settings
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
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int = 2,
        deterministic: bool = False,
        benchmark: bool = True,
        # Callbacks
        enable_checkpointing: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_monitor: str = "val/total_loss",
        checkpoint_mode: str = "min",
        save_top_k: int = 5,
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
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.num_sanity_val_steps = num_sanity_val_steps
        self.deterministic = deterministic
        self.benchmark = benchmark

        # Callbacks
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.save_top_k = save_top_k
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
            filename="hydraulic-gnn-{epoch:02d}-{val_total_loss:.4f}",
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_top_k=config.save_top_k,
            save_last=True,
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

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=strategy,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
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
    max_epochs: int = 200,
    devices: int = 1,
    checkpoint_dir: str | Path = "checkpoints/production",
    logger_save_dir: str | Path = "logs/production",
) -> pl.Trainer:
    """Create production trainer with best practices.

    Args:
        max_epochs: Maximum training epochs
        devices: Number of devices (GPUs)
        checkpoint_dir: Checkpoint directory
        logger_save_dir: Log directory

    Returns:
        Production-ready Trainer
    """
    config = TrainerConfig(
        max_epochs=max_epochs,
        devices=devices,
        accelerator="gpu" if devices > 0 else "cpu",
        precision="16-mixed" if devices > 0 else "32",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        checkpoint_dir=checkpoint_dir,
        save_top_k=5,
        enable_early_stopping=True,
        early_stopping_patience=30,
        logger_save_dir=logger_save_dir,
        logger_name="hydraulic_gnn_production",
        benchmark=True,
    )
    return create_trainer(config)


def create_development_trainer(
    max_epochs: int = 10,
    devices: int = 1,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """Create development trainer for debugging.

    Args:
        max_epochs: Maximum epochs
        devices: Number of devices
        fast_dev_run: Run single batch for debugging

    Returns:
        Development Trainer
    """
    config = TrainerConfig(
        max_epochs=max_epochs,
        devices=devices,
        accelerator="gpu" if devices > 0 else "cpu",
        precision="32",
        enable_checkpointing=False,
        enable_early_stopping=False,
        logger_save_dir="logs/dev",
        logger_name="hydraulic_gnn_dev",
        fast_dev_run=fast_dev_run,
    )
    return create_trainer(config)
