"""PyTorch Lightning module for hydraulic GNN training (IMPROVED).

Improvements:
- Unified configure_optimizers return type (always dict)
- Loss weighting validation
- Batch field assertions
- Better error messages
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
from src.training.losses import FocalLoss, QuantileRULLoss, UncertaintyWeighting, WingLoss

logger = logging.getLogger(__name__)


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics.

    Wraps UniversalTemporalGNN with Lightning training infrastructure.
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
        use_focal_loss: bool = True,
        use_wing_loss: bool = True,
        use_quantile_rul: bool = True,
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

        # Loss functions
        self.graph_health_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.graph_degradation_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.graph_anomaly_loss = FocalLoss(gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()
        self.graph_rul_loss = QuantileRULLoss() if use_quantile_rul else nn.MSELoss()
        self.component_health_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.component_anomaly_loss = FocalLoss(gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()

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
            self.uncertainty_weighter = UncertaintyWeighting(num_tasks=6)

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
        self, outputs: dict[str, dict[str, torch.Tensor]], batch: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-level multi-task loss."""
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

        # Compute losses
        graph_health_loss = self.graph_health_loss(outputs["graph"]["health"], batch.y_graph_health)
        graph_degradation_loss = self.graph_degradation_loss(
            outputs["graph"]["degradation"], batch.y_graph_degradation
        )
        graph_anomaly_loss = self.graph_anomaly_loss(outputs["graph"]["anomaly"], batch.y_graph_anomaly)
        graph_rul_loss = self.graph_rul_loss(outputs["graph"]["rul"], batch.y_graph_rul)
        component_health_loss = self.component_health_loss(
            outputs["component"]["health"], batch.y_component_health
        )
        component_anomaly_loss = self.component_anomaly_loss(
            outputs["component"]["anomaly"], batch.y_component_anomaly
        )

        # Combine
        if self.loss_weighting == "fixed":
            total_loss = (
                self.loss_weights["graph_health"] * graph_health_loss
                + self.loss_weights["graph_degradation"] * graph_degradation_loss
                + self.loss_weights["graph_anomaly"] * graph_anomaly_loss
                + self.loss_weights["graph_rul"] * graph_rul_loss
                + self.loss_weights["component_health"] * component_health_loss
                + self.loss_weights["component_anomaly"] * component_anomaly_loss
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
            total_loss = self.uncertainty_weighter(losses)

        loss_dict = {
            "graph_health": graph_health_loss,
            "graph_degradation": graph_degradation_loss,
            "graph_anomaly": graph_anomaly_loss,
            "graph_rul": graph_rul_loss,
            "component_health": component_health_loss,
            "component_anomaly": component_anomaly_loss,
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
        """Configure optimizers and schedulers (UNIFIED RETURN TYPE)."""
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
