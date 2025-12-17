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
    ConfidenceWeightedLoss,
    DomainAdversarialLoss,
    PhysicsAwareFocalLoss,
)

logger = logging.getLogger(__name__)


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics (Production-ready).

    Features:
    - Advanced physics-aware losses
    - Confidence-weighted training for imputed data
    - Domain adversarial loss for transfer learning
    - Multi-task uncertainty weighting
    - Manual optimization for multi-head output compatibility

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

        # CRITICAL: Enable manual optimization for multi-head compatibility
        self.automatic_optimization = False

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

            # Graph-level losses (reduction='mean' for scalars)
            self.graph_health_loss = nn.MSELoss(reduction="mean")
            self.graph_degradation_loss = nn.MSELoss(reduction="mean")
            self.graph_anomaly_loss = PhysicsAwareFocalLoss(
                alpha=0.25,
                gamma=2.0,
                severity_weights=severity_weights,
                reduction="mean",
            )
            self.graph_rul_loss = AsymmetricL1Loss(tau=rul_tau, reduction="mean")

            # Component-level losses
            self.component_health_loss = nn.MSELoss(reduction="mean")
            self.component_anomaly_loss = PhysicsAwareFocalLoss(
                alpha=0.25,
                gamma=2.0,
                component_weights=component_weights,
                severity_weights=severity_weights,
                reduction="mean",
            )

            # Wrap with confidence weighting if enabled
            if use_confidence_weighting:
                logger.info("Enabling confidence-weighted training")
                # Re-create with reduction='none' for confidence weighting
                self.graph_health_loss = ConfidenceWeightedLoss(
                    nn.MSELoss(reduction="none")
                )
                self.graph_degradation_loss = ConfidenceWeightedLoss(
                    nn.MSELoss(reduction="none")
                )
                self.graph_rul_loss = ConfidenceWeightedLoss(
                    AsymmetricL1Loss(tau=rul_tau, reduction="none")
                )
                self.component_health_loss = ConfidenceWeightedLoss(
                    nn.MSELoss(reduction="none")
                )

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
            # Use fixed weights (can be customized)
            init_weights = [1.0] * num_tasks
            self.uncertainty_weighter = UncertaintyWeighting(
                num_tasks=num_tasks,
                init_weights=init_weights
            )

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

        # CRITICAL: Squeeze all targets to prevent broadcasting issues
        # Convert [N, 1] -> [N] for MSE losses
        y_graph_health = batch.y_graph_health.squeeze(-1) if batch.y_graph_health.dim() > 1 else batch.y_graph_health
        y_graph_degradation = batch.y_graph_degradation.squeeze(-1) if batch.y_graph_degradation.dim() > 1 else batch.y_graph_degradation
        y_graph_rul = batch.y_graph_rul.squeeze(-1) if batch.y_graph_rul.dim() > 1 else batch.y_graph_rul
        y_component_health = batch.y_component_health.squeeze(-1) if batch.y_component_health.dim() > 1 else batch.y_component_health

        # Fix anomaly target shapes if flattened by DataLoader
        # Graph anomaly: should be [batch_size, 9]
        y_graph_anomaly = batch.y_graph_anomaly
        if y_graph_anomaly.dim() == 1:
            batch_size = outputs["graph"]["anomaly"].size(0)
            num_classes = outputs["graph"]["anomaly"].size(1)
            y_graph_anomaly = y_graph_anomaly.view(batch_size, num_classes)
        
        # Component anomaly: should be [num_nodes, 9]
        y_component_anomaly = batch.y_component_anomaly
        if y_component_anomaly.dim() == 1:
            num_nodes = outputs["component"]["anomaly"].size(0)
            num_classes = outputs["component"]["anomaly"].size(1)
            y_component_anomaly = y_component_anomaly.view(num_nodes, num_classes)

        # Get confidence if using confidence-weighted losses
        if self.use_confidence_weighting and self.use_advanced_losses:
            if not hasattr(batch, "confidence"):
                logger.warning("Confidence not found in batch, using 1.0")
                node_confidence = torch.ones(
                    batch.x.shape[0],
                    device=batch.x.device,
                )
                graph_confidence = torch.ones(
                    batch.num_graphs,
                    device=batch.x.device,
                )
            else:
                # Node-level confidence [num_nodes] e.g. [320]
                node_confidence = batch.confidence
                
                # Aggregate to graph-level confidence [num_graphs] e.g. [32]
                # Using native PyTorch scatter_reduce (mean)
                graph_confidence = torch.zeros(
                    batch.num_graphs, 
                    dtype=node_confidence.dtype, 
                    device=node_confidence.device
                )
                graph_confidence.scatter_reduce_(
                    dim=0,
                    index=batch.batch,
                    src=node_confidence,
                    reduce="mean",
                    include_self=False
                )

            # === GRAPH-LEVEL LOSSES (use graph_confidence) ===
            graph_health_loss = self.graph_health_loss(
                outputs["graph"]["health"].squeeze(-1),  # [batch_size]
                y_graph_health,
                graph_confidence
            )
            graph_degradation_loss = self.graph_degradation_loss(
                outputs["graph"]["degradation"].squeeze(-1),
                y_graph_degradation,
                graph_confidence
            )
            graph_rul_loss = self.graph_rul_loss(
                outputs["graph"]["rul"].squeeze(-1),
                y_graph_rul,
                graph_confidence
            )

            # === COMPONENT-LEVEL LOSSES (use node_confidence) ===
            component_health_loss = self.component_health_loss(
                outputs["component"]["health"].squeeze(-1),  # [num_nodes]
                y_component_health,
                node_confidence
            )

            # Anomaly losses (no confidence for classification)
            graph_anomaly_loss = self.graph_anomaly_loss(
                outputs["graph"]["anomaly"], 
                y_graph_anomaly  # Fixed shape
            )
            component_anomaly_loss = self.component_anomaly_loss(
                outputs["component"]["anomaly"], 
                y_component_anomaly  # Fixed shape
            )

        else:
            # Standard losses without confidence
            graph_health_loss = self.graph_health_loss(
                outputs["graph"]["health"].squeeze(-1), 
                y_graph_health
            )
            graph_degradation_loss = self.graph_degradation_loss(
                outputs["graph"]["degradation"].squeeze(-1), 
                y_graph_degradation
            )
            graph_anomaly_loss = self.graph_anomaly_loss(
                outputs["graph"]["anomaly"], 
                y_graph_anomaly  # Fixed shape
            )
            graph_rul_loss = self.graph_rul_loss(
                outputs["graph"]["rul"].squeeze(-1), 
                y_graph_rul
            )
            component_health_loss = self.component_health_loss(
                outputs["component"]["health"].squeeze(-1), 
                y_component_health
            )
            component_anomaly_loss = self.component_anomaly_loss(
                outputs["component"]["anomaly"], 
                y_component_anomaly  # Fixed shape
            )

        # Domain adversarial loss (optional)
        if self.use_domain_adversarial:
            if not hasattr(batch, "domain_labels"):
                logger.warning("Domain labels not found, skipping domain loss")
                domain_loss = torch.tensor(0.0, device=graph_health_loss.device)
            else:
                # Extract features from model
                features = outputs.get("features")
                if features is None:
                    logger.warning("Features not in outputs, skipping domain loss")
                    domain_loss = torch.tensor(0.0, device=graph_health_loss.device)
                else:
                    domain_pred = self.domain_classifier(features.detach())
                    domain_loss = self.domain_loss(domain_pred, batch.domain_labels)
        else:
            domain_loss = torch.tensor(0.0, device=graph_health_loss.device, dtype=torch.float32)

        # Ensure all losses are scalars + convert to float32 for stability
        def ensure_scalar_float32(loss: torch.Tensor) -> torch.Tensor:
            """Convert loss to scalar float32."""
            if loss.dim() > 0:
                loss = loss.mean()
            return loss.float()
        
        graph_health_loss = ensure_scalar_float32(graph_health_loss)
        graph_degradation_loss = ensure_scalar_float32(graph_degradation_loss)
        graph_anomaly_loss = ensure_scalar_float32(graph_anomaly_loss)
        graph_rul_loss = ensure_scalar_float32(graph_rul_loss)
        component_health_loss = ensure_scalar_float32(component_health_loss)
        component_anomaly_loss = ensure_scalar_float32(component_anomaly_loss)
        domain_loss = ensure_scalar_float32(domain_loss)

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
            # All losses are float32 scalars
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
            
            # UncertaintyWeighting now uses fixed buffers (no gradients)
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

    def training_step(self, batch: Any, _batch_idx: int) -> None:
        """Training step with manual backward."""
        # Get optimizer
        optimizer = self.optimizers()
        
        # Forward pass
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        # Manual backward
        self.manual_backward(total_loss, retain_graph=False)
        
        # Gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=batch.num_graphs)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"train/{key}_loss", val, prog_bar=False, batch_size=batch.num_graphs)

    def validation_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        self.log("val/total_loss", total_loss, prog_bar=True, batch_size=batch.num_graphs)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"val/{key}_loss", val, prog_bar=False, batch_size=batch.num_graphs)

        return total_loss

    def test_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch
        )
        total_loss, loss_dict = self.compute_loss(outputs, batch)

        self.log("test/total_loss", total_loss, batch_size=batch.num_graphs)
        for key, val in loss_dict.items():
            if key != "total":
                self.log(f"test/{key}_loss", val, batch_size=batch.num_graphs)

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type == "plateau":
            # Note: 'verbose' parameter removed in PyTorch 2.9+
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
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
