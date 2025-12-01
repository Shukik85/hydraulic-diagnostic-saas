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
    WingLoss,
    QuantileRULLoss,
    UncertaintyWeighting,
    MultiTaskLoss,
)
from .metrics import (
    MultiLevelMetrics,
    RegressionMetrics,
    ClassificationMetrics,
    RULMetrics,
    MetricConfig,
    create_metrics,
)
from .trainer import (
    create_trainer,
    create_development_trainer,
    create_production_trainer,
    TrainerConfig,
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggerConfig,
)

__all__ = [
    # Lightning module
    "HydraulicGNNModule",
    
    # Loss functions
    "FocalLoss",
    "WingLoss",
    "QuantileRULLoss",
    "UncertaintyWeighting",
    "MultiTaskLoss",
    
    # Metrics
    "MultiLevelMetrics",
    "RegressionMetrics",
    "ClassificationMetrics",
    "RULMetrics",
    "MetricConfig",
    "create_metrics",
    
    # Trainer
    "create_trainer",
    "create_development_trainer",
    "create_production_trainer",
    "TrainerConfig",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "LoggerConfig",
]
