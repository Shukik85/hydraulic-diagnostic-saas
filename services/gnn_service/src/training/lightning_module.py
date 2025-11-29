"""PyTorch Lightning module for hydraulic GNN training.

LightningModule wrapper for UniversalTemporalGNN with:
- Multi-level predictions (component + graph)
- Multi-task learning (health, degradation, anomaly, RUL)
- Advanced loss functions (Focal, Wing, Quantile)
- Uncertainty weighting
- Automatic optimization
- Metric tracking

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.models import UniversalTemporalGNN
from .losses import FocalLoss, WingLoss, QuantileRULLoss, UncertaintyWeighting


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics.
    
    Wraps UniversalTemporalGNN with Lightning training infrastructure:
    - Automatic optimization
    - Multi-level loss computation (component + graph)
    - Multi-task learning (health, degradation, anomaly, RUL)
    - Advanced losses (Focal, Wing, Quantile)
    - Metric tracking
    - Learning rate scheduling
    
    Attributes:
        model: UniversalTemporalGNN instance
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization
        scheduler_type: LR scheduler type
        loss_weighting: Loss weighting strategy
    
    Examples:
        >>> module = HydraulicGNNModule(
        ...     in_channels=34,
        ...     hidden_channels=128,
        ...     num_heads=8,
        ...     learning_rate=0.001,
        ...     loss_weighting="uncertainty"
        ... )
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(module, train_loader, val_loader)
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
        loss_weights: Dict[str, float] | None = None,
        use_focal_loss: bool = True,
        use_wing_loss: bool = True,
        use_quantile_rul: bool = True,
        **kwargs
    ):
        """Initialize Lightning module.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            num_heads: Number of attention heads
            num_gat_layers: Number of GAT layers
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            learning_rate: Learning rate
            weight_decay: L2 regularization
            scheduler_type: LR scheduler (plateau/cosine/none)
            loss_weighting: Weighting strategy (fixed/uncertainty)
            loss_weights: Task loss weights (if fixed)
            use_focal_loss: Use FocalLoss for anomaly
            use_wing_loss: Use WingLoss for regression
            use_quantile_rul: Use QuantileRULLoss for RUL
            **kwargs: Additional model arguments
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model
        self.model = UniversalTemporalGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_gat_layers=num_gat_layers,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            use_compile=False,  # Disable for training
            **kwargs
        )
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.loss_weighting = loss_weighting
        
        # === Graph-Level Loss Functions ===
        self.graph_health_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.graph_degradation_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.graph_anomaly_loss = FocalLoss(gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()
        self.graph_rul_loss = QuantileRULLoss() if use_quantile_rul else nn.MSELoss()
        
        # === Component-Level Loss Functions ===
        self.component_health_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        self.component_anomaly_loss = FocalLoss(gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()
        
        # === Multi-Task Weighting ===
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
            # 6 tasks: 4 graph + 2 component
            self.uncertainty_weighter = UncertaintyWeighting(num_tasks=6)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass with multi-level predictions.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 8]
            batch: Batch assignment [N]
        
        Returns:
            Nested dict:
            {
                'component': {'health': [N, 1], 'anomaly': [N, 9]},
                'graph': {'health': [B, 1], 'degradation': [B, 1], 
                          'anomaly': [B, 9], 'rul': [B, 1]}
            }
        """
        return self.model(x, edge_index, edge_attr, batch)
    
    def compute_loss(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-level multi-task loss.
        
        Args:
            outputs: Model outputs (nested dict)
            batch: Batch data with targets
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # === Graph-Level Losses ===
        graph_health_loss = self.graph_health_loss(
            outputs['graph']['health'],
            batch.y_graph_health
        )
        
        graph_degradation_loss = self.graph_degradation_loss(
            outputs['graph']['degradation'],
            batch.y_graph_degradation
        )
        
        graph_anomaly_loss = self.graph_anomaly_loss(
            outputs['graph']['anomaly'],
            batch.y_graph_anomaly
        )
        
        graph_rul_loss = self.graph_rul_loss(
            outputs['graph']['rul'],
            batch.y_graph_rul
        )
        
        # === Component-Level Losses ===
        component_health_loss = self.component_health_loss(
            outputs['component']['health'],
            batch.y_component_health
        )
        
        component_anomaly_loss = self.component_anomaly_loss(
            outputs['component']['anomaly'],
            batch.y_component_anomaly
        )
        
        # === Combine Losses ===
        if self.loss_weighting == "fixed":
            total_loss = (
                self.loss_weights["graph_health"] * graph_health_loss +
                self.loss_weights["graph_degradation"] * graph_degradation_loss +
                self.loss_weights["graph_anomaly"] * graph_anomaly_loss +
                self.loss_weights["graph_rul"] * graph_rul_loss +
                self.loss_weights["component_health"] * component_health_loss +
                self.loss_weights["component_anomaly"] * component_anomaly_loss
            )
        elif self.loss_weighting == "uncertainty":
            losses = {
                "graph_health": graph_health_loss,
                "graph_degradation": graph_degradation_loss,
                "graph_anomaly": graph_anomaly_loss,
                "graph_rul": graph_rul_loss,
                "component_health": component_health_loss,
                "component_anomaly": component_anomaly_loss,
            }
            total_loss = self.uncertainty_weighter(losses)
        
        # Loss dict for logging
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
    
    def training_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch from DataLoader
            batch_idx: Batch index
        
        Returns:
            loss: Total loss
        """
        # Forward pass
        outputs = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log("train/graph_health_loss", loss_dict["graph_health"], prog_bar=False)
        self.log("train/graph_degradation_loss", loss_dict["graph_degradation"], prog_bar=False)
        self.log("train/graph_anomaly_loss", loss_dict["graph_anomaly"], prog_bar=False)
        self.log("train/graph_rul_loss", loss_dict["graph_rul"], prog_bar=False)
        self.log("train/component_health_loss", loss_dict["component_health"], prog_bar=False)
        self.log("train/component_anomaly_loss", loss_dict["component_anomaly"], prog_bar=False)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        # Log uncertainty weights if using uncertainty weighting
        if self.loss_weighting == "uncertainty":
            log_vars = self.uncertainty_weighter.log_vars
            self.log("train/weight_graph_health", torch.exp(-log_vars[0]))
            self.log("train/weight_graph_degradation", torch.exp(-log_vars[1]))
            self.log("train/weight_graph_anomaly", torch.exp(-log_vars[2]))
            self.log("train/weight_graph_rul", torch.exp(-log_vars[3]))
            self.log("train/weight_component_health", torch.exp(-log_vars[4]))
            self.log("train/weight_component_anomaly", torch.exp(-log_vars[5]))
        
        return total_loss
    
    def validation_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch from DataLoader
            batch_idx: Batch index
        
        Returns:
            loss: Total loss
        """
        # Forward pass
        outputs = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log("val/graph_health_loss", loss_dict["graph_health"], prog_bar=False)
        self.log("val/graph_degradation_loss", loss_dict["graph_degradation"], prog_bar=False)
        self.log("val/graph_anomaly_loss", loss_dict["graph_anomaly"], prog_bar=False)
        self.log("val/graph_rul_loss", loss_dict["graph_rul"], prog_bar=False)
        self.log("val/component_health_loss", loss_dict["component_health"], prog_bar=False)
        self.log("val/component_anomaly_loss", loss_dict["component_anomaly"], prog_bar=False)
        self.log("val/total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def test_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> torch.Tensor:
        """Test step.
        
        Args:
            batch: Batch from DataLoader
            batch_idx: Batch index
        
        Returns:
            loss: Total loss
        """
        # Forward pass
        outputs = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log("test/graph_health_loss", loss_dict["graph_health"])
        self.log("test/graph_degradation_loss", loss_dict["graph_degradation"])
        self.log("test/graph_anomaly_loss", loss_dict["graph_anomaly"])
        self.log("test/graph_rul_loss", loss_dict["graph_rul"])
        self.log("test/component_health_loss", loss_dict["component_health"])
        self.log("test/component_anomaly_loss", loss_dict["component_anomaly"])
        self.log("test/total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.
        
        Returns:
            config: Optimizer and scheduler configuration
        """
        # Optimizer
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        if self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=1e-6
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        
        else:
            # No scheduler
            return {"optimizer": optimizer}
