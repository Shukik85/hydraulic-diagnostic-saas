"""Production-grade metrics for multi-level hydraulic GNN predictions.

Comprehensive metrics system for Universal Temporal GNN with:
- Multi-level tracking (component + graph)
- Multi-task metrics (health, degradation, anomaly, RUL)
- Regression metrics (MAE, RMSE, R²)
- Classification metrics (F1, Precision, Recall, AUC)
- RUL-specific metrics (horizon accuracy)
- PyTorch Lightning integration

Python 3.14 Features:
    - Deferred annotations
    - Union types
    - Improved dataclass performance
"""

from __future__ import annotations

from typing import Dict, Literal, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchmetrics import (
    Metric,
    MetricCollection,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    Precision,
    Recall,
    F1Score,
    AUROC,
)


@dataclass
class MetricConfig:
    """Configuration for metric computation.
    
    Attributes:
        num_anomaly_classes: Number of anomaly types (9 for hydraulics)
        rul_horizons: RUL prediction horizons in hours
        component_average: Averaging method for component metrics
        anomaly_average: Averaging method for anomaly metrics
        anomaly_threshold: Threshold for binary anomaly detection
    """
    num_anomaly_classes: int = 9
    rul_horizons: list[int] = field(default_factory=lambda: [24, 72, 168])  # 1d, 3d, 1w
    component_average: Literal["micro", "macro"] = "macro"
    anomaly_average: Literal["micro", "macro", "weighted"] = "macro"
    anomaly_threshold: float = 0.5


class RegressionMetrics(Metric):
    """Metrics for regression tasks (health, degradation, RUL).
    
    Computes:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)
    
    Attributes:
        prefix: Metric name prefix (e.g., "graph_health_")
        
    Examples:
        >>> metrics = RegressionMetrics(prefix="graph_health_")
        >>> metrics.update(preds, targets)
        >>> result = metrics.compute()  # {"graph_health_mae": 0.05, ...}
    """
    
    def __init__(
        self,
        prefix: str = "",
        **kwargs
    ):
        """Initialize regression metrics.
        
        Args:
            prefix: Metric name prefix
            **kwargs: Additional Metric arguments
        """
        super().__init__(**kwargs)
        
        self.prefix = prefix
        
        # Register metric components
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError(squared=False)  # RMSE
        self.r2 = R2Score()
        
        # For MAPE computation
        self.add_state("sum_ape", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """Update metrics with batch predictions.
        
        Args:
            preds: Predictions [B, 1] or [N, 1]
            target: Ground truth [B, 1] or [N, 1]
        """
        # Flatten tensors
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        
        # Update torchmetrics
        self.mae.update(preds_flat, target_flat)
        self.mse.update(preds_flat, target_flat)
        self.r2.update(preds_flat, target_flat)
        
        # Update MAPE state (avoid division by zero)
        epsilon = 1e-8
        ape = torch.abs((target_flat - preds_flat) / (target_flat + epsilon))
        self.sum_ape += ape.sum()
        self.total += target_flat.numel()
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics.
        
        Returns:
            Metric dictionary with prefix
        """
        mae_val = self.mae.compute()
        rmse_val = self.mse.compute()
        r2_val = self.r2.compute()
        mape_val = self.sum_ape / self.total if self.total > 0 else torch.tensor(0.0)
        
        return {
            f"{self.prefix}mae": mae_val,
            f"{self.prefix}rmse": rmse_val,
            f"{self.prefix}r2": r2_val,
            f"{self.prefix}mape": mape_val,
        }
    
    def reset(self) -> None:
        """Reset all metric states."""
        self.mae.reset()
        self.mse.reset()
        self.r2.reset()
        self.sum_ape = torch.tensor(0.0)
        self.total = torch.tensor(0)


class ClassificationMetrics(Metric):
    """Metrics for multi-label classification (anomaly detection).
    
    Computes:
        - Precision (per-class and averaged)
        - Recall (per-class and averaged)
        - F1 Score (per-class and averaged)
        - AUC-ROC (per-class and averaged)
    
    Attributes:
        num_classes: Number of anomaly types
        average: Averaging method (micro/macro/weighted)
        threshold: Binary classification threshold
        prefix: Metric name prefix
        
    Examples:
        >>> metrics = ClassificationMetrics(
        ...     num_classes=9,
        ...     average="macro",
        ...     prefix="component_anomaly_"
        ... )
        >>> metrics.update(logits, targets)
        >>> result = metrics.compute()
    """
    
    def __init__(
        self,
        num_classes: int,
        average: Literal["micro", "macro", "weighted"] = "macro",
        threshold: float = 0.5,
        prefix: str = "",
        **kwargs
    ):
        """Initialize classification metrics.
        
        Args:
            num_classes: Number of classes
            average: Averaging method
            threshold: Binary threshold
            prefix: Metric name prefix
            **kwargs: Additional Metric arguments
        """
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.prefix = prefix
        
        # Multi-label classification task
        task = "multilabel"
        
        # Register metrics
        self.precision = Precision(
            task=task,
            num_labels=num_classes,
            average=average,
            threshold=threshold
        )
        
        self.recall = Recall(
            task=task,
            num_labels=num_classes,
            average=average,
            threshold=threshold
        )
        
        self.f1 = F1Score(
            task=task,
            num_labels=num_classes,
            average=average,
            threshold=threshold
        )
        
        self.auroc = AUROC(
            task=task,
            num_labels=num_classes,
            average=average
        )
    
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """Update metrics with batch predictions.
        
        Args:
            preds: Logits [B, C] or [N, C]
            target: Binary labels [B, C] or [N, C]
        """
        # Convert logits to probabilities for AUROC
        probs = torch.sigmoid(preds)
        
        # Update all metrics
        self.precision.update(preds, target.long())
        self.recall.update(preds, target.long())
        self.f1.update(preds, target.long())
        self.auroc.update(probs, target.long())
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics.
        
        Returns:
            Metric dictionary with prefix
        """
        precision_val = self.precision.compute()
        recall_val = self.recall.compute()
        f1_val = self.f1.compute()
        auroc_val = self.auroc.compute()
        
        return {
            f"{self.prefix}precision": precision_val,
            f"{self.prefix}recall": recall_val,
            f"{self.prefix}f1": f1_val,
            f"{self.prefix}auroc": auroc_val,
        }
    
    def reset(self) -> None:
        """Reset all metric states."""
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()


class RULMetrics(Metric):
    """Specialized metrics for Remaining Useful Life prediction.
    
    Computes:
        - Horizon Accuracy: % predictions within ±h hours
        - Asymmetric Loss: Penalizes late predictions more
        - Mean Horizon Error: Error at specific horizons
    
    Attributes:
        horizons: Prediction horizons in hours [24, 72, 168]
        prefix: Metric name prefix
        
    Examples:
        >>> metrics = RULMetrics(horizons=[24, 72, 168], prefix="graph_rul_")
        >>> metrics.update(rul_preds, rul_targets)
        >>> result = metrics.compute()  # Includes horizon-specific accuracy
    """
    
    def __init__(
        self,
        horizons: list[int] | None = None,
        prefix: str = "",
        **kwargs
    ):
        """Initialize RUL metrics.
        
        Args:
            horizons: Time horizons in hours
            prefix: Metric name prefix
            **kwargs: Additional Metric arguments
        """
        super().__init__(**kwargs)
        
        self.horizons = horizons or [24, 72, 168]
        self.prefix = prefix
        
        # Base regression metrics
        self.mae = MeanAbsoluteError()
        
        # Horizon accuracy states
        for horizon in self.horizons:
            self.add_state(
                f"correct_h{horizon}",
                default=torch.tensor(0),
                dist_reduce_fx="sum"
            )
        
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("asymmetric_loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """Update RUL metrics.
        
        Args:
            preds: RUL predictions [B, 1]
            target: True RUL [B, 1]
        """
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        
        # Base MAE
        self.mae.update(preds_flat, target_flat)
        
        # Error
        error = preds_flat - target_flat
        
        # Horizon accuracy: within ±h hours
        for horizon in self.horizons:
            correct = (torch.abs(error) <= horizon).sum()
            getattr(self, f"correct_h{horizon}").add_(correct)
        
        # Asymmetric loss (penalize late predictions more)
        # L = |e| if e >= 0 (early/on-time), 2|e| if e < 0 (late)
        asymmetric = torch.where(
            error >= 0,
            torch.abs(error),
            2 * torch.abs(error)
        )
        self.asymmetric_loss_sum += asymmetric.sum()
        
        self.total += target_flat.numel()
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute RUL metrics.
        
        Returns:
            Metric dictionary including horizon accuracies
        """
        mae_val = self.mae.compute()
        
        # Horizon accuracy
        horizon_acc = {}
        for horizon in self.horizons:
            correct = getattr(self, f"correct_h{horizon}")
            accuracy = correct.float() / self.total if self.total > 0 else torch.tensor(0.0)
            horizon_acc[f"{self.prefix}acc_h{horizon}"] = accuracy
        
        # Asymmetric loss
        asymmetric_val = self.asymmetric_loss_sum / self.total if self.total > 0 else torch.tensor(0.0)
        
        return {
            f"{self.prefix}mae": mae_val,
            f"{self.prefix}asymmetric_loss": asymmetric_val,
            **horizon_acc,
        }
    
    def reset(self) -> None:
        """Reset all metric states."""
        self.mae.reset()
        
        for horizon in self.horizons:
            getattr(self, f"correct_h{horizon}").zero_()
        
        self.total.zero_()
        self.asymmetric_loss_sum.zero_()


class MultiLevelMetrics:
    """Unified metrics for multi-level predictions (component + graph).
    
    Manages all metrics for Universal Temporal GNN:
        - Component-level: health (regression), anomaly (classification)
        - Graph-level: health, degradation (regression), anomaly (classification), RUL (RUL metrics)
    
    Integrates with PyTorch Lightning for automatic logging.
    
    Attributes:
        config: Metric configuration
        stage: Training stage (train/val/test)
        
    Examples:
        >>> config = MetricConfig(num_anomaly_classes=9)
        >>> metrics = MultiLevelMetrics(config, stage="val")
        >>> 
        >>> # During validation
        >>> metrics.update(outputs, batch)
        >>> result = metrics.compute()  # All metrics
        >>> metrics.reset()
    """
    
    def __init__(
        self,
        config: MetricConfig | None = None,
        stage: Literal["train", "val", "test"] = "train"
    ):
        """Initialize multi-level metrics.
        
        Args:
            config: Metric configuration
            stage: Training stage for logging prefix
        """
        self.config = config or MetricConfig()
        self.stage = stage
        
        # === Component-Level Metrics ===
        self.component_health_metrics = RegressionMetrics(
            prefix=f"{stage}/component_health_"
        )
        
        self.component_anomaly_metrics = ClassificationMetrics(
            num_classes=self.config.num_anomaly_classes,
            average=self.config.component_average,
            threshold=self.config.anomaly_threshold,
            prefix=f"{stage}/component_anomaly_"
        )
        
        # === Graph-Level Metrics ===
        self.graph_health_metrics = RegressionMetrics(
            prefix=f"{stage}/graph_health_"
        )
        
        self.graph_degradation_metrics = RegressionMetrics(
            prefix=f"{stage}/graph_degradation_"
        )
        
        self.graph_anomaly_metrics = ClassificationMetrics(
            num_classes=self.config.num_anomaly_classes,
            average=self.config.anomaly_average,
            threshold=self.config.anomaly_threshold,
            prefix=f"{stage}/graph_anomaly_"
        )
        
        self.graph_rul_metrics = RULMetrics(
            horizons=self.config.rul_horizons,
            prefix=f"{stage}/graph_rul_"
        )
    
    def update(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        batch: Any
    ) -> None:
        """Update all metrics with batch predictions.
        
        Args:
            outputs: Model outputs (nested dict)
                {
                    'component': {'health': [N,1], 'anomaly': [N,9]},
                    'graph': {'health': [B,1], 'degradation': [B,1], 
                              'anomaly': [B,9], 'rul': [B,1]}
                }
            batch: Batch with ground truth targets
        """
        # === Component-Level Updates ===
        self.component_health_metrics.update(
            outputs['component']['health'],
            batch.y_component_health
        )
        
        self.component_anomaly_metrics.update(
            outputs['component']['anomaly'],
            batch.y_component_anomaly
        )
        
        # === Graph-Level Updates ===
        self.graph_health_metrics.update(
            outputs['graph']['health'],
            batch.y_graph_health
        )
        
        self.graph_degradation_metrics.update(
            outputs['graph']['degradation'],
            batch.y_graph_degradation
        )
        
        self.graph_anomaly_metrics.update(
            outputs['graph']['anomaly'],
            batch.y_graph_anomaly
        )
        
        self.graph_rul_metrics.update(
            outputs['graph']['rul'],
            batch.y_graph_rul
        )
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all metrics.
        
        Returns:
            Flat dictionary with all metrics
        """
        metrics = {}
        
        # Component-level
        metrics.update(self.component_health_metrics.compute())
        metrics.update(self.component_anomaly_metrics.compute())
        
        # Graph-level
        metrics.update(self.graph_health_metrics.compute())
        metrics.update(self.graph_degradation_metrics.compute())
        metrics.update(self.graph_anomaly_metrics.compute())
        metrics.update(self.graph_rul_metrics.compute())
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metric states."""
        # Component-level
        self.component_health_metrics.reset()
        self.component_anomaly_metrics.reset()
        
        # Graph-level
        self.graph_health_metrics.reset()
        self.graph_degradation_metrics.reset()
        self.graph_anomaly_metrics.reset()
        self.graph_rul_metrics.reset()
    
    def log_dict(self) -> Dict[str, float]:
        """Get metrics as float dict for Lightning logging.
        
        Returns:
            Metric dictionary with float values
        """
        metrics = self.compute()
        return {k: v.item() for k, v in metrics.items()}


def create_metrics(
    stage: Literal["train", "val", "test"],
    num_anomaly_classes: int = 9,
    rul_horizons: list[int] | None = None
) -> MultiLevelMetrics:
    """Factory for creating multi-level metrics.
    
    Args:
        stage: Training stage
        num_anomaly_classes: Number of anomaly types
        rul_horizons: RUL prediction horizons
    
    Returns:
        Configured MultiLevelMetrics instance
        
    Examples:
        >>> val_metrics = create_metrics("val", num_anomaly_classes=9)
        >>> test_metrics = create_metrics("test", rul_horizons=[24, 72, 168])
    """
    config = MetricConfig(
        num_anomaly_classes=num_anomaly_classes,
        rul_horizons=rul_horizons or [24, 72, 168]
    )
    
    return MultiLevelMetrics(config=config, stage=stage)
