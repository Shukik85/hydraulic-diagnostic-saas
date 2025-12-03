"""Production-ready PyTorch Lightning Trainer factory.

Production-grade training infrastructure with:
- Trainer factory with sensible defaults
- ModelCheckpoint callback (save best models)
- EarlyStopping callback (prevent overfitting)
- TensorBoard logger (metrics visualization)
- Gradient clipping (training stability)
- Mixed precision training (FP16)
- Development and production configs

Python 3.14 Features:
    - Deferred annotations
    - Union types
    - Pattern matching for config
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
    """Configuration for PyTorch Lightning Trainer.
    
    Attributes:
        max_epochs: Maximum training epochs
        accelerator: Device type (gpu/cpu/mps)
        devices: Number of devices (1 for single-GPU)
        precision: Training precision (32/16/bf16)
        gradient_clip_val: Gradient clipping threshold
        accumulate_grad_batches: Gradient accumulation steps
        log_every_n_steps: Logging frequency
        val_check_interval: Validation frequency
        deterministic: Reproducibility mode
        enable_checkpointing: Enable checkpoint saving
        enable_progress_bar: Show progress bar
        enable_model_summary: Show model summary
        strategy: Distributed strategy
    """
    # Training duration
    max_epochs: int = 100

    # Hardware
    accelerator: Literal["gpu", "cpu", "mps"] = "gpu"
    devices: int = 1
    precision: Literal["32", "16", "bf16"] = "16"

    # Optimization
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Every epoch

    # Reproducibility
    deterministic: bool = False

    # Features
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True

    # Distributed
    strategy: str | DDPStrategy = "auto"


@dataclass
class CheckpointConfig:
    """Configuration for ModelCheckpoint callback.
    
    Attributes:
        dirpath: Checkpoint save directory
        filename: Checkpoint filename pattern
        monitor: Metric to monitor
        mode: Optimization mode (min/max)
        save_top_k: Number of best models to keep
        save_last: Always save last checkpoint
        every_n_epochs: Save every N epochs
        verbose: Print save messages
    """
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
    """Configuration for EarlyStopping callback.
    
    Attributes:
        monitor: Metric to monitor
        patience: Epochs without improvement before stopping
        mode: Optimization mode (min/max)
        min_delta: Minimum change to qualify as improvement
        verbose: Print stopping messages
        strict: Raise error if metric not found
    """
    monitor: str = "val/total_loss"
    patience: int = 20
    mode: Literal["min", "max"] = "min"
    min_delta: float = 1e-4
    verbose: bool = True
    strict: bool = True


@dataclass
class LoggerConfig:
    """Configuration for logging.
    
    Attributes:
        save_dir: Log save directory
        name: Experiment name
        version: Experiment version
        log_graph: Log model graph
        default_hp_metric: Default hyperparameter metric
    """
    save_dir: str | Path = "logs"
    name: str = "hydraulic_gnn"
    version: str | None = None
    log_graph: bool = True
    default_hp_metric: bool = False


def create_checkpoint_callback(
    config: CheckpointConfig | None = None
) -> ModelCheckpoint:
    """Create ModelCheckpoint callback.
    
    Saves best models based on validation loss.
    
    Args:
        config: Checkpoint configuration
    
    Returns:
        Configured ModelCheckpoint callback
        
    Examples:
        >>> checkpoint = create_checkpoint_callback()
        >>> trainer = pl.Trainer(callbacks=[checkpoint])
    """
    config = config or CheckpointConfig()

    # Ensure directory exists
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


def create_early_stopping_callback(
    config: EarlyStoppingConfig | None = None
) -> EarlyStopping:
    """Create EarlyStopping callback.
    
    Stops training when validation metric stops improving.
    
    Args:
        config: Early stopping configuration
    
    Returns:
        Configured EarlyStopping callback
        
    Examples:
        >>> early_stop = create_early_stopping_callback()
        >>> trainer = pl.Trainer(callbacks=[early_stop])
    """
    config = config or EarlyStoppingConfig()

    return EarlyStopping(
        monitor=config.monitor,
        patience=config.patience,
        mode=config.mode,
        min_delta=config.min_delta,
        verbose=config.verbose,
        strict=config.strict,
    )


def create_tensorboard_logger(
    config: LoggerConfig | None = None
) -> TensorBoardLogger:
    """Create TensorBoard logger.
    
    Logs metrics, hyperparameters, and model graph.
    
    Args:
        config: Logger configuration
    
    Returns:
        Configured TensorBoard logger
        
    Examples:
        >>> logger = create_tensorboard_logger()
        >>> trainer = pl.Trainer(logger=logger)
        >>> # View logs: tensorboard --logdir=logs
    """
    config = config or LoggerConfig()

    # Ensure directory exists
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
    """Factory for creating production-ready Lightning Trainer.
    
    Creates fully configured Trainer with:
        - ModelCheckpoint callback
        - EarlyStopping callback
        - TensorBoard logger
        - CSV logger (backup)
        - Learning rate monitor
        - Rich progress bar
        - Gradient clipping
        - Mixed precision (FP16)
    
    Args:
        trainer_config: Trainer configuration
        checkpoint_config: Checkpoint configuration
        early_stopping_config: Early stopping configuration
        logger_config: Logger configuration
        additional_callbacks: Additional callbacks
        fast_dev_run: Run single batch for debugging
    
    Returns:
        Configured Lightning Trainer
        
    Examples:
        >>> # Development trainer (fast iteration)
        >>> dev_trainer = create_trainer(fast_dev_run=True)
        >>> 
        >>> # Production trainer (full training)
        >>> prod_config = TrainerConfig(
        ...     max_epochs=200,
        ...     precision="16",
        ...     devices=2
        ... )
        >>> prod_trainer = create_trainer(trainer_config=prod_config)
        >>> 
        >>> # Custom checkpoint location
        >>> checkpoint_cfg = CheckpointConfig(
        ...     dirpath="models/checkpoints",
        ...     save_top_k=5
        ... )
        >>> custom_trainer = create_trainer(checkpoint_config=checkpoint_cfg)
    """
    # === Configurations ===
    trainer_config = trainer_config or TrainerConfig()
    checkpoint_config = checkpoint_config or CheckpointConfig()
    early_stopping_config = early_stopping_config or EarlyStoppingConfig()
    logger_config = logger_config or LoggerConfig()

    # === Callbacks ===
    callbacks = []

    # ModelCheckpoint
    if trainer_config.enable_checkpointing:
        callbacks.append(create_checkpoint_callback(checkpoint_config))

    # EarlyStopping
    callbacks.append(create_early_stopping_callback(early_stopping_config))

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Progress bar (if enabled)
    if trainer_config.enable_progress_bar:
        callbacks.append(RichProgressBar())

    # Model summary (if enabled)
    if trainer_config.enable_model_summary:
        callbacks.append(RichModelSummary(max_depth=2))

    # Additional callbacks
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

    # === Distributed Strategy ===
    strategy = trainer_config.strategy

    # If using multiple GPUs, configure DDP
    if isinstance(trainer_config.devices, int) and trainer_config.devices > 1:
        if strategy == "auto":
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

    # === Create Trainer ===
    trainer = pl.Trainer(
        # Duration
        max_epochs=trainer_config.max_epochs,

        # Hardware
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        strategy=strategy,

        # Optimization
        gradient_clip_val=trainer_config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,

        # Validation
        check_val_every_n_epoch=1,
        val_check_interval=trainer_config.val_check_interval,

        # Logging
        log_every_n_steps=trainer_config.log_every_n_steps,
        logger=loggers,

        # Callbacks
        callbacks=callbacks,
        enable_checkpointing=trainer_config.enable_checkpointing,
        enable_progress_bar=trainer_config.enable_progress_bar,
        enable_model_summary=trainer_config.enable_model_summary,

        # Reproducibility
        deterministic=trainer_config.deterministic,

        # Development
        fast_dev_run=fast_dev_run,

        # Performance
        benchmark=True,
        inference_mode=True,  # Faster validation
    )

    return trainer


def create_development_trainer() -> pl.Trainer:
    """Create trainer optimized for development.
    
    Fast iteration with:
        - Lower precision (FP16)
        - Frequent logging
        - Short patience
        - Rich progress bar
    
    Returns:
        Development-optimized Trainer
        
    Examples:
        >>> trainer = create_development_trainer()
        >>> trainer.fit(module, train_loader, val_loader)
    """
    trainer_config = TrainerConfig(
        max_epochs=50,
        precision="16",
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    early_stopping_config = EarlyStoppingConfig(
        patience=10,  # Shorter patience
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


def create_production_trainer(
    max_epochs: int = 200,
    devices: int = 1
) -> pl.Trainer:
    """Create trainer optimized for production training.
    
    Full training with:
        - High precision or mixed precision
        - Long patience
        - Extensive checkpointing
        - Multi-GPU support
    
    Args:
        max_epochs: Maximum training epochs
        devices: Number of GPUs
    
    Returns:
        Production-optimized Trainer
        
    Examples:
        >>> trainer = create_production_trainer(max_epochs=300, devices=2)
        >>> trainer.fit(module, train_loader, val_loader)
    """
    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        devices=devices,
        precision="16",  # Mixed precision for speed
        log_every_n_steps=50,
        accumulate_grad_batches=1,
    )

    early_stopping_config = EarlyStoppingConfig(
        patience=30,  # Longer patience
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
