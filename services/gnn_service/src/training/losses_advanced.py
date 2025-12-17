"""Advanced loss functions for hydraulic diagnostics.

Implements physics-aware and asymmetric losses:
- AsymmetricL1Loss for RUL (penalize underestimation)
- QuantileRULLoss (multi-quantile prediction)
- DomainAdversarialLoss (DIDA pattern)
- PhysicsAwareFocalLoss

References:
  [1] Quantile RUL loss: arXiv:2311.17410
  [2] DIDA: Domain-invariant diagnosis
  [3] Focal loss with physics weighting
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AsymmetricL1Loss(nn.Module):
    """Asymmetric L1 loss for RUL prediction.

    Penalizes underestimation (predicting failure too late) more heavily
    than overestimation (predicting failure too early).

    Args:
        tau: Asymmetry parameter (0.5 = symmetric, 0.7 = 70% penalty for underestimation)
        reduction: 'mean', 'sum', or 'none'

    Examples:
        >>> loss_fn = AsymmetricL1Loss(tau=0.7)
        >>> pred = torch.tensor([100.0])  # Predicted RUL
        >>> true = torch.tensor([50.0])   # True RUL
        >>> loss = loss_fn(pred, true)
        >>> # Underestimation: loss = 0.7 * |100-50| = 35.0
    """

    def __init__(self, tau: float = 0.7, reduction: str = "mean"):
        super().__init__()
        if not 0 < tau < 1:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric L1 loss.

        Args:
            pred: Predicted RUL [N, 1]
            target: True RUL [N, 1]

        Returns:
            Loss value
        """
        error = pred - target

        # Asymmetric weighting
        # If error > 0 (overestimation): weight = (1 - tau)
        # If error < 0 (underestimation): weight = tau
        weight = torch.where(error > 0, 1.0 - self.tau, self.tau)

        loss = weight * torch.abs(error)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class QuantileRULLoss(nn.Module):
    """Multi-quantile RUL loss.

    Predicts multiple quantiles (e.g., 0.1, 0.5, 0.9) to capture uncertainty.
    Based on pinball loss / quantile regression.

    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        reduction: 'mean', 'sum', or 'none'

    Examples:
        >>> loss_fn = QuantileRULLoss(quantiles=[0.1, 0.5, 0.9])
        >>> pred = torch.randn(32, 3)  # 3 quantiles
        >>> true = torch.randn(32, 1)  # True RUL
        >>> loss = loss_fn(pred, true)
    """

    def __init__(self, quantiles: list[float] = [0.1, 0.5, 0.9], reduction: str = "mean"):
        super().__init__()
        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantile must be in (0, 1), got {q}")
        self.quantiles = torch.tensor(quantiles)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss (pinball loss).

        Args:
            pred: Predicted quantiles [N, Q]
            target: True RUL [N, 1]

        Returns:
            Loss value
        """
        quantiles = self.quantiles.to(pred.device)
        target = target.expand_as(pred)  # [N, Q]

        error = pred - target  # [N, Q]

        # Pinball loss
        loss = torch.where(
            error > 0,
            quantiles * error,  # Overestimation
            (quantiles - 1) * error,  # Underestimation
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PhysicsAwareFocalLoss(nn.Module):
    """Focal loss with physics-based importance weighting.

    Assigns higher weights to critical components (e.g., pumps, valves)
    and failure modes with high severity.

    Args:
        alpha: Class balancing factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        component_weights: Importance weights per component [N]
        severity_weights: Severity weights per anomaly class [C]
        reduction: 'mean', 'sum', or 'none'

    Examples:
        >>> loss_fn = PhysicsAwareFocalLoss(
        ...     alpha=0.25,
        ...     gamma=2.0,
        ...     component_weights=torch.tensor([1.0, 0.8, 0.6]),  # Pump > valve > sensor
        ...     severity_weights=torch.tensor([1.0, 0.9, 0.8, ...]),  # 9 anomaly classes
        ... )
        >>> logits = torch.randn(3, 9)  # 3 nodes, 9 anomaly classes
        >>> targets = torch.randint(0, 2, (3, 9)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        component_weights: torch.Tensor | None = None,
        severity_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.component_weights = component_weights
        self.severity_weights = severity_weights
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute physics-aware focal loss.

        Args:
            logits: Predicted logits [N, C]
            targets: True labels [N, C]

        Returns:
            Loss value
        """
        probs = torch.sigmoid(logits)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal term
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal loss
        focal_loss = alpha_t * focal_term * bce

        # Component importance weighting
        if self.component_weights is not None:
            comp_weights = self.component_weights.to(logits.device)
            comp_weights = comp_weights.view(-1, 1).expand_as(focal_loss)
            focal_loss = focal_loss * comp_weights

        # Severity weighting
        if self.severity_weights is not None:
            sev_weights = self.severity_weights.to(logits.device)
            sev_weights = sev_weights.view(1, -1).expand_as(focal_loss)
            focal_loss = focal_loss * sev_weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss for domain-invariant features (DIDA pattern).

    Encourages feature extractor to learn domain-invariant representations
    by fooling a domain classifier.

    Args:
        lambda_domain: Trade-off parameter (typically 0.1-1.0)
        num_domains: Number of domains (e.g., 2 for source/target)

    Examples:
        >>> # In training loop:
        >>> feature_extractor = nn.Sequential(...)
        >>> domain_classifier = nn.Sequential(...)
        >>> da_loss = DomainAdversarialLoss(lambda_domain=0.5, num_domains=2)
        >>>
        >>> # Extract features
        >>> features = feature_extractor(x)
        >>>
        >>> # Domain classification
        >>> domain_pred = domain_classifier(features.detach())  # Detach for adversarial
        >>> domain_loss = da_loss(domain_pred, domain_labels)
        >>>
        >>> # Total loss
        >>> total_loss = task_loss - domain_loss  # Maximize domain confusion
    """

    def __init__(self, lambda_domain: float = 0.5, num_domains: int = 2):
        super().__init__()
        self.lambda_domain = lambda_domain
        self.num_domains = num_domains
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_pred: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """Compute domain adversarial loss.

        Args:
            domain_pred: Domain predictions [N, D]
            domain_labels: True domain labels [N]

        Returns:
            Loss value (negative for gradient reversal)
        """
        domain_loss = self.criterion(domain_pred, domain_labels)
        return self.lambda_domain * domain_loss


class ConfidenceWeightedLoss(nn.Module):
    """Wrapper for confidence-weighted loss.

    Reduces loss contribution from low-confidence predictions
    (e.g., from imputed sensors).

    Args:
        base_loss: Base loss function (e.g., MSELoss, FocalLoss)
        min_confidence: Minimum confidence to use (default 0.1)

    Examples:
        >>> base_loss = nn.MSELoss(reduction='none')
        >>> conf_loss = ConfidenceWeightedLoss(base_loss, min_confidence=0.1)
        >>> pred = torch.randn(32, 1)
        >>> target = torch.randn(32, 1)
        >>> confidence = torch.rand(32, 1)  # From imputation
        >>> loss = conf_loss(pred, target, confidence)
    """

    def __init__(self, base_loss: nn.Module, min_confidence: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.min_confidence = min_confidence

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence-weighted loss.

        Args:
            pred: Predictions
            target: Targets
            confidence: Confidence scores [N] or [N, 1]

        Returns:
            Weighted loss
        """
        # Compute base loss (no reduction)
        loss = self.base_loss(pred, target)

        # Clamp confidence
        confidence = torch.clamp(confidence, min=self.min_confidence, max=1.0)

        # Weight by confidence
        weighted_loss = loss * confidence.expand_as(loss)

        return weighted_loss.mean()
