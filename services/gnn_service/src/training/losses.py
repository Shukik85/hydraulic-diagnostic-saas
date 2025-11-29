"""Loss functions for multi-task hydraulic diagnostics.

Custom losses:
- FocalLoss - Handles class imbalance in anomaly detection
- WingLoss - Robust regression for health/degradation
- UncertaintyWeighting - Dynamic multi-task weighting
- QuantileRULLoss - Asymmetric RUL prediction
- MultiTaskLoss - Combined loss wrapper

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focuses learning on hard examples by down-weighting easy ones.
    Useful for anomaly detection where normal cases dominate.
    
    Args:
        alpha: Class weights [C] or scalar
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Loss reduction (mean/sum/none)
    
    Examples:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 9)  # [B, C]
        >>> targets = torch.randint(0, 2, (32, 9)).float()  # [B, C]
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        alpha: float | torch.Tensor = 0.25,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Predicted logits [B, C]
            targets: Binary targets [B, C]
        
        Returns:
            loss: Focal loss scalar or [B, C]
        """
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )
        
        # Focal term: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if isinstance(self.alpha, (int, float)):
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha
        
        # Focal loss
        loss = alpha_t * focal_term * bce
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WingLoss(nn.Module):
    """Wing Loss for robust regression.
    
    Combines L1 and L2 losses with smooth transition.
    More robust to outliers than MSE, less biased than MAE.
    
    Args:
        omega: Threshold for transition between L1/L2
        epsilon: Smoothness parameter
        reduction: Loss reduction
    
    References:
        - Wing Loss: https://arxiv.org/abs/1711.06753
    
    Examples:
        >>> loss_fn = WingLoss(omega=10.0, epsilon=2.0)
        >>> pred = torch.randn(32, 1)
        >>> target = torch.randn(32, 1)
        >>> loss = loss_fn(pred, target)
    """
    
    def __init__(
        self,
        omega: float = 10.0,
        epsilon: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Precompute constant C
        self.C = self.omega - self.omega * torch.log(
            torch.tensor(1.0 + self.omega / self.epsilon)
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute wing loss.
        
        Args:
            pred: Predictions [B, 1]
            target: Targets [B, 1]
        
        Returns:
            loss: Wing loss scalar or [B, 1]
        """
        delta = torch.abs(pred - target)
        
        # Wing loss formula
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class QuantileRULLoss(nn.Module):
    """Quantile loss for RUL (Remaining Useful Life) prediction.
    
    Asymmetric loss that penalizes underestimation (predicting too early failure)
    more heavily than overestimation (predicting too late failure).
    
    This is critical for maintenance planning where:
    - Underestimating RUL → unexpected failure (costly, dangerous)
    - Overestimating RUL → early maintenance (less costly, safer)
    
    Args:
        quantiles: List of quantiles to optimize (default: [0.1, 0.5, 0.9])
        reduction: Loss reduction strategy
    
    References:
        - Quantile Regression for RUL: https://openreview.net/forum?id=tzFjcVqmxw
        - Multi-Task ST-GNN: https://arxiv.org/pdf/2401.15964.pdf
    
    Examples:
        >>> loss_fn = QuantileRULLoss(quantiles=[0.1, 0.5, 0.9])
        >>> pred = torch.randn(32, 1).abs() * 100  # Predicted hours
        >>> target = torch.randn(32, 1).abs() * 100  # True hours
        >>> loss = loss_fn(pred, target)
        
        # With log-normalization (recommended)
        >>> pred_log = torch.log1p(pred)
        >>> target_log = torch.log1p(target)
        >>> loss = loss_fn(pred_log, target_log)
    """
    
    def __init__(
        self,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        super().__init__()
        
        # Validate quantiles
        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
        
        self.quantiles = quantiles
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantile loss.
        
        Args:
            pred: Predicted RUL [B, 1] (can be log-normalized)
            target: True RUL [B, 1] (can be log-normalized)
        
        Returns:
            loss: Quantile loss scalar or [B, 1]
        
        Formula:
            For each quantile q:
                L_q = max(q * (target - pred), (q - 1) * (target - pred))
            
            Total loss = mean over all quantiles
        
        Effect:
            - If pred < target (underestimation):
                penalty = q * error
            - If pred > target (overestimation):
                penalty = (1 - q) * error
            
            With q = 0.9:
                underestimation penalty = 0.9 * error
                overestimation penalty = 0.1 * error
                → Underestimation penalized 9x more!
        """
        losses = []
        
        for q in self.quantiles:
            # Error: positive if underestimated, negative if overestimated
            error = target - pred
            
            # Quantile loss (asymmetric)
            quantile_loss = torch.max(
                q * error,
                (q - 1) * error
            )
            
            losses.append(quantile_loss)
        
        # Stack and average over quantiles
        stacked_losses = torch.stack(losses, dim=0)  # [num_quantiles, B, 1]
        loss = stacked_losses.mean(dim=0)  # [B, 1]
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class UncertaintyWeighting(nn.Module):
    """Uncertainty-based multi-task weighting.
    
    Learns task weights dynamically based on homoscedastic uncertainty.
    Balances tasks automatically without manual tuning.
    
    Args:
        num_tasks: Number of tasks
        init_log_var: Initial log variance for each task
    
    References:
        - Multi-Task Learning Using Uncertainty: https://arxiv.org/abs/1705.07115
    
    Examples:
        >>> weighter = UncertaintyWeighting(num_tasks=3)
        >>> losses = {
        ...     "health": torch.tensor(0.5),
        ...     "degradation": torch.tensor(0.3),
        ...     "anomaly": torch.tensor(0.8)
        ... }
        >>> total_loss = weighter(losses)
    """
    
    def __init__(
        self,
        num_tasks: int,
        init_log_var: float = 0.0
    ):
        super().__init__()
        
        # Learnable log variances (one per task)
        self.log_vars = nn.Parameter(
            torch.full((num_tasks,), init_log_var)
        )
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute weighted loss.
        
        Args:
            losses: Dictionary of task losses
        
        Returns:
            total_loss: Weighted sum of losses
        
        Examples:
            Formula: L_total = sum_i (exp(-log_var_i) * L_i + log_var_i)
        """
        task_names = list(losses.keys())
        
        total_loss = 0.0
        
        for i, task_name in enumerate(task_names):
            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            loss_weighted = precision * losses[task_name] + self.log_vars[i]
            
            total_loss = total_loss + loss_weighted
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss wrapper.
    
    Combines health, degradation, and anomaly losses with configurable weighting.
    
    Args:
        health_loss: Loss for health regression
        degradation_loss: Loss for degradation regression
        anomaly_loss: Loss for anomaly classification
        weighting: Weighting strategy (fixed/uncertainty)
        loss_weights: Fixed weights (if weighting='fixed')
    
    Examples:
        >>> multi_loss = MultiTaskLoss(
        ...     health_loss=nn.MSELoss(),
        ...     degradation_loss=WingLoss(),
        ...     anomaly_loss=FocalLoss(),
        ...     weighting="uncertainty"
        ... )
        >>> loss = multi_loss(health_pred, degradation_pred, anomaly_logits,
        ...                   health_true, degradation_true, anomaly_true)
    """
    
    def __init__(
        self,
        health_loss: nn.Module | None = None,
        degradation_loss: nn.Module | None = None,
        anomaly_loss: nn.Module | None = None,
        weighting: Literal["fixed", "uncertainty"] = "fixed",
        loss_weights: Dict[str, float] | None = None
    ):
        super().__init__()
        
        # Loss functions
        self.health_loss = health_loss or nn.MSELoss()
        self.degradation_loss = degradation_loss or nn.MSELoss()
        self.anomaly_loss = anomaly_loss or nn.BCEWithLogitsLoss()
        
        # Weighting strategy
        self.weighting = weighting
        
        if weighting == "fixed":
            self.loss_weights = loss_weights or {
                "health": 1.0,
                "degradation": 1.0,
                "anomaly": 1.0
            }
        elif weighting == "uncertainty":
            self.uncertainty_weighter = UncertaintyWeighting(num_tasks=3)
    
    def forward(
        self,
        health_pred: torch.Tensor,
        degradation_pred: torch.Tensor,
        anomaly_logits: torch.Tensor,
        health_true: torch.Tensor,
        degradation_true: torch.Tensor,
        anomaly_true: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss.
        
        Args:
            health_pred: Health predictions [B, 1]
            degradation_pred: Degradation predictions [B, 1]
            anomaly_logits: Anomaly logits [B, 9]
            health_true: Health targets [B, 1]
            degradation_true: Degradation targets [B, 1]
            anomaly_true: Anomaly targets [B, 9]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # Compute individual losses
        health_loss = self.health_loss(health_pred, health_true)
        degradation_loss = self.degradation_loss(
            degradation_pred,
            degradation_true
        )
        anomaly_loss = self.anomaly_loss(anomaly_logits, anomaly_true)
        
        # Combine losses
        if self.weighting == "fixed":
            total_loss = (
                self.loss_weights["health"] * health_loss +
                self.loss_weights["degradation"] * degradation_loss +
                self.loss_weights["anomaly"] * anomaly_loss
            )
        elif self.weighting == "uncertainty":
            losses = {
                "health": health_loss,
                "degradation": degradation_loss,
                "anomaly": anomaly_loss
            }
            total_loss = self.uncertainty_weighter(losses)
        
        # Return total + individual losses
        loss_dict = {
            "health": health_loss,
            "degradation": degradation_loss,
            "anomaly": anomaly_loss,
            "total": total_loss
        }
        
        return total_loss, loss_dict
