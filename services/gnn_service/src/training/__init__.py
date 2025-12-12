"""Training module exports.

Complete training infrastructure:
    - LightningModule wrapper
    - Loss functions (Focal, Wing, Quantile, Uncertainty)
    - Metrics (Multi-level, Regression, Classification, RUL)
    - Trainer factory (Development, Production)
"""

from .lightning_module import HydraulicGNNModule
from .losses import (
    FocalLoss,
    MultiTaskLoss,
    QuantileRULLoss,
    UncertaintyWeighting,
    WingLoss,
)
from .metrics import (
    ClassificationMetrics,
    MetricConfig,
    MultiLevelMetrics,
    RegressionMetrics,
    RULMetrics,
    create_metrics,
)
from .trainer import (
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggerConfig,
    TrainerConfig,
    create_development_trainer,
    create_production_trainer,
    create_trainer,
)

__all__ = [
    "CheckpointConfig",
    "ClassificationMetrics",
    "EarlyStoppingConfig",
    # Loss functions
    "FocalLoss",
    # Lightning module
    "HydraulicGNNModule",
    "LoggerConfig",
    "MetricConfig",
    # Metrics
    "MultiLevelMetrics",
    "MultiTaskLoss",
    "QuantileRULLoss",
    "RULMetrics",
    "RegressionMetrics",
    "TrainerConfig",
    "UncertaintyWeighting",
    "WingLoss",
    "create_development_trainer",
    "create_metrics",
    "create_production_trainer",
    # Trainer
    "create_trainer",
]
