"""Production-ready PyTorch Lightning Trainer factory (IMPROVED).

Improvements:
- Better DDPStrategy initialization logic
- Fast dev run config adjustment
- Cleaner strategy selection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning Trainer."""

    max_epochs: int = 100
    accelerator: Literal["gpu", "cpu", "mps"] = "gpu"
    devices: int = 1
    precision: Literal["32", "16", "bf16"] = "16"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    deterministic: bool = False
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    strategy: str | DDPStrategy = "auto"


@dataclass
class CheckpointConfig:
    """Configuration for ModelCheckpoint callback."""

    dirpath: str | Path = "checkpoints"
    filename: str = "hydraulic-gnn-{epoch:02d}-{val_total_loss:.4f}"
    monitor: str = "val/total_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    verbose: bool = True


@dataclass
class EarlyStoppingConfig:
    """Configuration for EarlyStopping callback."""

    monitor: str = "val/total_loss"
    patience: int = 20
    mode: Literal["min", "max"] = "min"
    min_delta: float = 1e-4
    verbose: bool = True
    strict: bool = True


@dataclass
class LoggerConfig:
    """Configuration for logging."""

    save_dir: str | Path = "logs"
    name: str = "hydraulic_gnn"
    version: str | None = None
    log_graph: bool = True
    default_hp_metric: bool = False


def create_checkpoint_callback(config: CheckpointConfig | None = None) -> ModelCheckpoint:
    """Create ModelCheckpoint callback."""
    config = config or CheckpointConfig()
    dirpath = Path(config.dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    return ModelCheckpoint(
        dirpath=str(dirpath),
        filename=config.filename,
        monitor=config.monitor,
        mode=config.mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last,
        every_n_epochs=config.every_n_epochs,
        verbose=config.verbose,
        auto_insert_metric_name=False,
    )


def create_early_stopping_callback(config: EarlyStoppingConfig | None = None) -> EarlyStopping:
    """Create EarlyStopping callback."""
    config = config or EarlyStoppingConfig()

    return EarlyStopping(
        monitor=config.monitor,
        patience=config.patience,
        mode=config.mode,
        min_delta=config.min_delta,
        verbose=config.verbose,
        strict=config.strict,
    )


def create_tensorboard_logger(config: LoggerConfig | None = None) -> TensorBoardLogger:
    """Create TensorBoard logger."""
    config = config or LoggerConfig()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return TensorBoardLogger(
        save_dir=str(save_dir),
        name=config.name,
        version=config.version,
        log_graph=config.log_graph,
        default_hp_metric=config.default_hp_metric,
    )


def create_trainer(
    trainer_config: TrainerConfig | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    early_stopping_config: EarlyStoppingConfig | None = None,
    logger_config: LoggerConfig | None = None,
    additional_callbacks: list | None = None,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """Factory for creating production-ready Lightning Trainer (IMPROVED)."""
    trainer_config = trainer_config or TrainerConfig()
    checkpoint_config = checkpoint_config or CheckpointConfig()
    early_stopping_config = early_stopping_config or EarlyStoppingConfig()
    logger_config = logger_config or LoggerConfig()

    # Adjust for fast_dev_run
    if fast_dev_run:
        trainer_config.max_epochs = 1
        trainer_config.val_check_interval = 0.1
        trainer_config.log_every_n_steps = 1

    # === Callbacks ===
    callbacks = []

    if trainer_config.enable_checkpointing:
        callbacks.append(create_checkpoint_callback(checkpoint_config))

    callbacks.append(create_early_stopping_callback(early_stopping_config))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if trainer_config.enable_progress_bar:
        callbacks.append(RichProgressBar())

    if trainer_config.enable_model_summary:
        callbacks.append(RichModelSummary(max_depth=2))

    if additional_callbacks:
        callbacks.extend(additional_callbacks)

    # === Loggers ===
    loggers = [
        create_tensorboard_logger(logger_config),
        CSVLogger(
            save_dir=str(logger_config.save_dir),
            name=f"{logger_config.name}_csv",
        ),
    ]

    # === Distributed Strategy (IMPROVED) ===
    strategy = trainer_config.strategy

    if isinstance(trainer_config.devices, int) and trainer_config.devices > 1:
        if strategy == "auto":
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

    # === Create Trainer ===
    return pl.Trainer(
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        strategy=strategy,
        gradient_clip_val=trainer_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        val_check_interval=trainer_config.val_check_interval,
        log_every_n_steps=trainer_config.log_every_n_steps,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=trainer_config.enable_checkpointing,
        enable_progress_bar=trainer_config.enable_progress_bar,
        enable_model_summary=trainer_config.enable_model_summary,
        deterministic=trainer_config.deterministic,
        fast_dev_run=fast_dev_run,
        benchmark=True,
        inference_mode=True,
    )


def create_development_trainer() -> pl.Trainer:
    """Create trainer optimized for development."""
    trainer_config = TrainerConfig(
        max_epochs=50,
        precision="16",
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    early_stopping_config = EarlyStoppingConfig(
        patience=10,
        verbose=True,
    )

    checkpoint_config = CheckpointConfig(
        dirpath="checkpoints/dev",
        save_top_k=2,
    )

    return create_trainer(
        trainer_config=trainer_config,
        checkpoint_config=checkpoint_config,
        early_stopping_config=early_stopping_config,
    )


def create_production_trainer(max_epochs: int = 200, devices: int = 1) -> pl.Trainer:
    """Create trainer optimized for production training."""
    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        devices=devices,
        precision="16",
        log_every_n_steps=50,
        accumulate_grad_batches=1,
    )

    early_stopping_config = EarlyStoppingConfig(
        patience=30,
        min_delta=1e-5,
    )

    checkpoint_config = CheckpointConfig(
        dirpath="checkpoints/production",
        save_top_k=5,
        save_last=True,
    )

    logger_config = LoggerConfig(
        save_dir="logs/production",
        name="hydraulic_gnn_prod",
    )

    return create_trainer(
        trainer_config=trainer_config,
        checkpoint_config=checkpoint_config,
        early_stopping_config=early_stopping_config,
        logger_config=logger_config,
    )
