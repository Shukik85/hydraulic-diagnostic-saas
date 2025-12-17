"""Training components for Universal Temporal GNN.

Provides:
- LightningModule for training
- Loss functions (standard + advanced)
- Metrics
- DataLoaders
- Imputation (GRAPE two-stage)
- Trainer factories
"""

__version__ = "0.2.0"

from src.training.lightning_module import HydraulicGNNModule
from src.training.trainer import (
    create_trainer,
    create_production_trainer,
    create_development_trainer,
)
from src.training.losses import (
    FocalLoss,
    WingLoss,
    QuantileRULLoss as QuantileRULLossBasic,
    UncertaintyWeighting,
    MultiTaskLoss,
)
from src.training.losses_advanced import (
    AsymmetricL1Loss,
    QuantileRULLoss,
    PhysicsAwareFocalLoss,
    DomainAdversarialLoss,
    ConfidenceWeightedLoss,
)
from src.training.metrics import (
    RULMetrics,
    HealthMetrics,
    AnomalyMetrics,
    create_metrics,
)
from src.training.imputation_grape import (
    GRAPEImputer,
    TemporalImputer,
    TwoStageImputer,
)
from src.training.imputation_engine import ImputationEngine
from src.training.graph_reconstructor import GraphReconstructor

__all__ = [
    # Training
    "HydraulicGNNModule",
    # Trainers
    "create_trainer",
    "create_production_trainer",
    "create_development_trainer",
    # Standard losses
    "FocalLoss",
    "WingLoss",
    "QuantileRULLossBasic",
    "UncertaintyWeighting",
    "MultiTaskLoss",
    # Advanced losses (GRAPE, DIDA, Quantile)
    "AsymmetricL1Loss",
    "QuantileRULLoss",
    "PhysicsAwareFocalLoss",
    "DomainAdversarialLoss",
    "ConfidenceWeightedLoss",
    # Metrics
    "RULMetrics",
    "HealthMetrics",
    "AnomalyMetrics",
    "create_metrics",
    # Imputation
    "GRAPEImputer",
    "TemporalImputer",
    "TwoStageImputer",
    "ImputationEngine",
    # Graph reconstruction
    "GraphReconstructor",
]
