"""PyTorch Lightning module for hydraulic GNN training (GRAPE + DIDA integrated).

Integrates:
- GRAPE two-stage imputation
- Advanced losses (AsymmetricL1, QuantileRUL, PhysicsAwareFocal)
- Confidence-weighted training
- Domain adversarial loss (DIDA)
- Component criticality weighting
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.models import UniversalTemporalGNN
from src.training.losses import UncertaintyWeighting
from src.training.losses_advanced import (
    AsymmetricL1Loss,
    PhysicsAwareFocalLoss,
    DomainAdversarialLoss,
    ConfidenceWeightedLoss,
)

logger = logging.getLogger(__name__)


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics (Production-ready).

    Features:
    - Advanced physics-aware losses
    - Confidence-weighted training for imputed data
    - Domain adversarial loss for transfer learning
    - Multi-task uncertainty weighting

    Examples:
        >>> module = HydraulicGNNModule(
        ...     in_channels=34,
        ...     hidden_channels=128,
        ...     use_advanced_losses=True,
        ...     use_confidence_weighting=True,
        ...     component_weights=torch.tensor([1.0, 0.8, 0.6]),
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_gat_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_type: Literal["plateau", "cosine", "none"] = "plateau",
        loss_weighting: Literal["fixed", "uncertainty"] = "fixed",
        loss_weights: dict[str, float] | None = None,
        # Advanced loss options
        use_advanced_losses: bool = True,
        use_confidence_weighting: bool = True,
        use_domain_adversarial: bool = False,
        # RUL loss config
        rul_tau: float = 0.7,  # AsymmetricL1 tau
        # Physics-aware config
        component_weights: torch.Tensor | None = None,
        severity_weights: torch.Tensor | None = None,
        # Domain adversarial config
        lambda_domain: float = 0.5,
        num_domains: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Validate loss_weighting
        if loss_weighting not in ["fixed", "uncertainty"]:
            raise ValueError(f"Unknown loss_weighting: {loss_weighting}")

        # Model
        self.model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_gat_layers=num_gat_layers,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            use_compile=False,
            **kwargs,
        )

        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.loss_weighting = loss_weighting
        self.use_advanced_losses = use_advanced_losses
        self.use_confidence_weighting = use_confidence_weighting
        self.use_domain_adversarial = use_domain_adversarial

        # === ADVANCED LOSS FUNCTIONS ===
        if use_advanced_losses:
            logger.info("Using advanced losses (GRAPE + DIDA patterns)")

            # Graph-level losses
            self.graph_health_loss = nn.MSELoss(reduction="none")  # Wrapped with confidence
            self.graph_degradation_loss = nn.MSELoss(reduction="none")
            self.graph_anomaly_loss = PhysicsAwareFocalLoss(
                alpha=0.25,
                gamma=2.0,
                severity_weights=severity_weights,
                reduction="none",
            )
            self.graph_rul_loss = AsymmetricL1Loss(tau=rul_tau, reduction="none")

            # Component-level losses
            self.component_health_loss = nn.MSELoss(reduction="none")
            self.component_anomaly_loss = PhysicsAwareFocalLoss(
                alpha=0.25,
                gamma=2.0,
                component_weights=component_weights,
                severity_weights=severity_weights,
                reduction="none",
            )

            # Wrap with confidence weighting if enabled
            if use_confidence_weighting:
                logger.info("Enabling confidence-weighted training")
                self.graph_health_loss = ConfidenceWeightedLoss(self.graph_health_loss)
                self.graph_degradation_loss = ConfidenceWeightedLoss(self.graph_degradation_loss)
                self.graph_rul_loss = ConfidenceWeightedLoss(self.graph_rul_loss)
                self.component_health_loss = ConfidenceWeightedLoss(self.component_health_loss)

        else:
            # Basic losses (backward compatibility)
            logger.info("Using basic losses")
            from src.training.losses import FocalLoss, WingLoss

            self.graph_health_loss = WingLoss()
            self.graph_degradation_loss = WingLoss()
            self.graph_anomaly_loss = FocalLoss(gamma=2.0)
            self.graph_rul_loss = nn.MSELoss()
            self.component_health_loss = WingLoss()
            self.component_anomaly_loss = FocalLoss(gamma=2.0)

        # Domain adversarial loss (DIDA)
        if use_domain_adversarial:
            logger.info("Enabling domain adversarial loss (DIDA)")
            self.domain_loss = DomainAdversarialLoss(
                lambda_domain=lambda_domain,
                num_domains=num_domains,
            )
            # Domain classifier (to be trained separately)
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_channels, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_domains),
            )

        # Multi-task weighting
        if loss_weighting == "fixed":
            self.loss_weights = loss_weights or {
                "graph_health": 1.0,
                "graph_degradation": 1.0,
                "graph_anomaly": 1.0,
                "graph_rul": 1.0,
                "component_health": 0.5,
                "component_anomaly": 0.5,
            }
        elif loss_weighting == "uncertainty":
            num_tasks = 7 if use_domain_adversarial else 6
            self.uncertainty_weighter = UncertaintyWeighting(num_tasks=num_tasks)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass."""
        return self.model(x, edge_index, edge_attr, batch)

    def compute_loss(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-level multi-task loss with confidence weighting.

        Args:
            outputs: Model outputs
            batch: Batch data (must contain confidence if use_confidence_weighting=True)

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Validate batch
        required_fields = [
            "y_graph_health",
            "y_graph_degradation",
            "y_graph_anomaly",
            "y_graph_rul",
            "y_component_health",
            "y_component_anomaly",
        ]
        for field in required_fields:
            if not hasattr(batch, field):
                raise AttributeError(f"Batch missing required field: {field}")

        # Get confidence if using confidence-weighted losses
        if self.use_confidence_weighting and self.use_advanced_losses:
            if not hasattr(batch, "confidence"):
                logger.warning("Confidence not found in batch, using 1.0")
                confidence = torch.ones(
                    batch.y_graph_health.shape[0],
                    device=batch.y_graph_health.device,
                )
            else:
                confidence = batch.confidence

            # Compute losses with confidence
            graph_health_loss = self.graph_health_loss(
                outputs["graph"]["health"], batch.y_graph_health, confidence
            )
            graph_degradation_loss = self.graph_degradation_loss(
                outputs["graph"]["degradation"], batch.y_graph_degradation, confidence
            )
            graph_rul_loss = self.graph_rul_loss(
                outputs["graph"]["rul"], batch.y_graph_rul, confidence
            )
            component_health_loss = self.component_health_loss(
                outputs["component"]["health"],
                batch.y_component_health,
                confidence.unsqueeze(-1).expand_as(batch.y_component_health),
            )

            # Anomaly losses (no confidence for classification)
            graph_anomaly_loss = self.graph_anomaly_loss(
                outputs["graph"]["anomaly"], batch.y_graph_anomaly
            )
            component_anomaly_loss = self.component_anomaly_loss(
                outputs["component"]["anomaly"], batch.y_component_anomaly
            )

        else:
            # Standard losses without confidence
            graph_health_loss = self.graph_health_loss(
                outputs["graph"]["health"], batch.y_graph_health
            )
            graph_degradation_loss = self.graph_degradation_loss(
                outputs["graph"]["degradation"], batch.y_graph_degradation
            )
            graph_anomaly_loss = self.graph_anomaly_loss(
                outputs["graph"]["anomaly"], batch.y_graph_anomaly
            )
            graph_rul_loss = self.graph_rul_loss(outputs["graph"]["rul"], batch.y_graph_rul)
            component_health_loss = self.component_health_loss(
                outputs["component"]["health"], batch.y_component_health
            )
            component_anomaly_loss = self.component_anomaly_loss(
                outputs["component"]["anomaly"], batch.y_component_anomaly
            )

        # Domain adversarial loss (optional)
        if self.use_domain_adversarial:
            if not hasattr(batch, "domain_labels"):
                logger.warning("Domain labels not found, skipping domain loss")
                domain_loss = torch.tensor(0.0, device=graph_health_loss.device)
            else:
                # Extract features from model
                features = outputs.get("features", None)
                if features is None:
                    logger.warning("Features not in outputs, skipping domain loss")
                    domain_loss = torch.tensor(0.0, device=graph_health_loss.device)
                else:
                    domain_pred = self.domain_classifier(features.detach())
                    domain_loss = self.domain_loss(domain_pred, batch.domain_labels)
        else:
            domain_loss = torch.tensor(0.0, device=graph_health_loss.device)

        # Combine losses
        if self.loss_weighting == "fixed":
            total_loss = (
                self.loss_weights["graph_health"] * graph_health_loss
                + self.loss_weights["graph_degradation"] * graph_degradation_loss
                + self.loss_weights["graph_anomaly"] * graph_anomaly_loss
                + self.loss_weights["graph_rul"] * graph_rul_loss
                + self.loss_weights["component_health"] * component_health_loss
                + self.loss_weights["component_anomaly"] * component_anomaly_loss
                + domain_loss  # DIDA
            )
        else:  # uncertainty
            losses = {
                "graph_health": graph_health_loss,
                "graph_degradation": graph_degradation_loss,
                "graph_anomaly": graph_anomaly_loss,
                "graph_rul": graph_rul_loss,
                "component_health": component_health_loss,
                "component_anomaly": component_anomaly_loss,
            }
            if self.use_domain_adversarial:
                losses["domain"] = domain_loss
            total_loss = self.uncertainty_weighter(losses)

        loss_dict = {
            "graph_health": graph_health_loss,
            "graph_degradation": graph_degradation_loss,
            "graph_anomaly": graph_anomaly_loss,
            "graph_rul": graph_rul_loss,
            "component_health": component_health_loss,
            "component_anomaly": component_anomaly_loss,
            "domain": domain_loss,
            "total": total_loss,
        }

        return total_loss, loss_dict

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        self.log("train/total_loss", total_loss, prog_bar=True)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"train/{key}_loss", val, prog_bar=False)

        if self.loss_weighting == "uncertainty":
            log_vars = self.uncertainty_weighter.log_vars
            for i, task in enumerate(
                [
                    "graph_health",
                    "graph_degradation",
                    "graph_anomaly",
                    "graph_rul",
                    "component_health",
                    "component_anomaly",
                ]
            ):
                self.log(f"train/weight_{task}", torch.exp(-log_vars[i]))

        return total_loss

    def validation_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        self.log("val/total_loss", total_loss, prog_bar=True)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"val/{key}_loss", val, prog_bar=False)

        return total_loss

    def test_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        self.log("test/total_loss", total_loss)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"test/{key}_loss", val)

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, verbose=False
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"},
            }
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return {"optimizer": optimizer}
