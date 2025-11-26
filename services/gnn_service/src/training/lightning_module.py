"""PyTorch Lightning module for hydraulic GNN training.

LightningModule wrapper for UniversalTemporalGNN with:
- Multi-task learning (health, degradation, anomaly)
- Automatic optimization
- Metric tracking
- Model checkpointing

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.models import UniversalTemporalGNN


class HydraulicGNNModule(pl.LightningModule):
    """PyTorch Lightning module for hydraulic diagnostics.
    
    Wraps UniversalTemporalGNN with Lightning training infrastructure:
    - Automatic optimization
    - Multi-task loss computation
    - Metric tracking
    - Learning rate scheduling
    
    Attributes:
        model: UniversalTemporalGNN instance
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization
        scheduler_type: LR scheduler type
        loss_weights: Multi-task loss weights
    
    Examples:
        >>> module = HydraulicGNNModule(
        ...     in_channels=34,
        ...     hidden_channels=128,
        ...     num_heads=8,
        ...     learning_rate=0.001
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
        scheduler_type: str = "plateau",
        loss_weights: Dict[str, float] | None = None,
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
            scheduler_type: LR scheduler (plateau/cosine)
            loss_weights: Task loss weights {health, degradation, anomaly}
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
            use_compile=False,  # Disable for training (gradient compatibility)
            **kwargs
        )
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        # Loss weights (default: equal weighting)
        self.loss_weights = loss_weights or {
            "health": 1.0,
            "degradation": 1.0,
            "anomaly": 1.0
        }
        
        # Loss functions (will be defined in losses.py)
        self.health_loss_fn = nn.MSELoss()
        self.degradation_loss_fn = nn.MSELoss()
        self.anomaly_loss_fn = nn.BCEWithLogitsLoss()
    
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
        
        # Compute losses
        health_loss = self.health_loss_fn(health_pred, batch.y_health)
        degradation_loss = self.degradation_loss_fn(
            degradation_pred,
            batch.y_degradation
        )
        anomaly_loss = self.anomaly_loss_fn(
            anomaly_logits,
            batch.y_anomaly
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights["health"] * health_loss +
            self.loss_weights["degradation"] * degradation_loss +
            self.loss_weights["anomaly"] * anomaly_loss
        )
        
        # Log metrics
        self.log("train/health_loss", health_loss, prog_bar=True)
        self.log("train/degradation_loss", degradation_loss)
        self.log("train/anomaly_loss", anomaly_loss)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
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
        
        # Compute losses
        health_loss = self.health_loss_fn(health_pred, batch.y_health)
        degradation_loss = self.degradation_loss_fn(
            degradation_pred,
            batch.y_degradation
        )
        anomaly_loss = self.anomaly_loss_fn(
            anomaly_logits,
            batch.y_anomaly
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights["health"] * health_loss +
            self.loss_weights["degradation"] * degradation_loss +
            self.loss_weights["anomaly"] * anomaly_loss
        )
        
        # Log metrics
        self.log("val/health_loss", health_loss, prog_bar=True)
        self.log("val/degradation_loss", degradation_loss)
        self.log("val/anomaly_loss", anomaly_loss)
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
        
        # Compute losses
        health_loss = self.health_loss_fn(health_pred, batch.y_health)
        degradation_loss = self.degradation_loss_fn(
            degradation_pred,
            batch.y_degradation
        )
        anomaly_loss = self.anomaly_loss_fn(
            anomaly_logits,
            batch.y_anomaly
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights["health"] * health_loss +
            self.loss_weights["degradation"] * degradation_loss +
            self.loss_weights["anomaly"] * anomaly_loss
        )
        
        # Log metrics
        self.log("test/health_loss", health_loss)
        self.log("test/degradation_loss", degradation_loss)
        self.log("test/anomaly_loss", anomaly_loss)
        self.log("test/total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.
        
        Returns:
            config: Optimizer and scheduler configuration
        
        Examples:
            >>> config = module.configure_optimizers()
            >>> optimizer = config["optimizer"]
            >>> scheduler = config["lr_scheduler"]
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
                T_max=100,  # Will be overridden by trainer max_epochs
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
