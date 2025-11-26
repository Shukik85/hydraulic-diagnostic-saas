"""PyTorch Lightning module for hydraulic GNN training.

LightningModule wrapper for UniversalTemporalGNN with:
- Multi-task learning (health, degradation, anomaly)
- Advanced loss functions (Focal, Wing)
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
from .losses import FocalLoss, WingLoss, MultiTaskLoss


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics.
    
    Wraps UniversalTemporalGNN with Lightning training infrastructure:
    - Automatic optimization
    - Multi-task loss computation
    - Advanced losses (Focal, Wing)
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
        
        # Loss functions
        health_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        degradation_loss = WingLoss() if use_wing_loss else nn.MSELoss()
        anomaly_loss = FocalLoss(gamma=2.0) if use_focal_loss else nn.BCEWithLogitsLoss()
        
        self.multi_task_loss = MultiTaskLoss(
            health_loss=health_loss,
            degradation_loss=degradation_loss,
            anomaly_loss=anomaly_loss,
            weighting=loss_weighting,
            loss_weights=loss_weights
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 8]
            batch: Batch assignment [N]
        
        Returns:
            health: Health predictions [B, 1]
            degradation: Degradation predictions [B, 1]
            anomaly: Anomaly logits [B, 9]
        """
        return self.model(x, edge_index, edge_attr, batch)
    
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
        health_pred, degradation_pred, anomaly_logits = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute multi-task loss
        total_loss, loss_dict = self.multi_task_loss(
            health_pred=health_pred,
            degradation_pred=degradation_pred,
            anomaly_logits=anomaly_logits,
            health_true=batch.y_health,
            degradation_true=batch.y_degradation,
            anomaly_true=batch.y_anomaly
        )
        
        # Log metrics
        self.log("train/health_loss", loss_dict["health"], prog_bar=False)
        self.log("train/degradation_loss", loss_dict["degradation"], prog_bar=False)
        self.log("train/anomaly_loss", loss_dict["anomaly"], prog_bar=False)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        # Log uncertainty weights if using uncertainty weighting
        if self.loss_weighting == "uncertainty":
            log_vars = self.multi_task_loss.uncertainty_weighter.log_vars
            self.log("train/weight_health", torch.exp(-log_vars[0]))
            self.log("train/weight_degradation", torch.exp(-log_vars[1]))
            self.log("train/weight_anomaly", torch.exp(-log_vars[2]))
        
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
        health_pred, degradation_pred, anomaly_logits = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute multi-task loss
        total_loss, loss_dict = self.multi_task_loss(
            health_pred=health_pred,
            degradation_pred=degradation_pred,
            anomaly_logits=anomaly_logits,
            health_true=batch.y_health,
            degradation_true=batch.y_degradation,
            anomaly_true=batch.y_anomaly
        )
        
        # Log metrics
        self.log("val/health_loss", loss_dict["health"], prog_bar=False)
        self.log("val/degradation_loss", loss_dict["degradation"], prog_bar=False)
        self.log("val/anomaly_loss", loss_dict["anomaly"], prog_bar=False)
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
        health_pred, degradation_pred, anomaly_logits = self(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        
        # Compute multi-task loss
        total_loss, loss_dict = self.multi_task_loss(
            health_pred=health_pred,
            degradation_pred=degradation_pred,
            anomaly_logits=anomaly_logits,
            health_true=batch.y_health,
            degradation_true=batch.y_degradation,
            anomaly_true=batch.y_anomaly
        )
        
        # Log metrics
        self.log("test/health_loss", loss_dict["health"])
        self.log("test/degradation_loss", loss_dict["degradation"])
        self.log("test/anomaly_loss", loss_dict["anomaly"])
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
