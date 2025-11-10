"""
Training script for Temporal GAT hydraulic diagnostics model.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from config import model_config, training_config
from dataset import HydraulicGraphDataset, split_dataset
from model import TemporalGAT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class HydraulicTrainer:
    """Trainer for hydraulic diagnostics GNN model."""

    def __init__(self, config: training_config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def setup_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders."""
        try:
            # Load full dataset
            dataset = HydraulicGraphDataset()
            logger.info(f"Loaded dataset with {len(dataset)} graphs")

            # Split dataset
            train_dataset, val_dataset, test_dataset = split_dataset(
                dataset,
                self.config.train_split,
                self.config.val_split,
                self.config.test_split,
            )

            # Create data loaders
            train_loader = train_dataset.get_data_loader(
                batch_size=self.config.batch_size, shuffle=True
            )
            val_loader = val_dataset.get_data_loader(
                batch_size=self.config.batch_size, shuffle=False
            )
            test_loader = test_dataset.get_data_loader(
                batch_size=self.config.batch_size, shuffle=False
            )

            logger.info(
                f"Data loaders created: "
                f"Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise

    def setup_model(self) -> TemporalGAT:
        """Initialize model."""
        model = TemporalGAT(
            num_node_features=model_config.num_node_features,
            hidden_dim=model_config.hidden_dim,
            num_classes=model_config.num_classes,
            num_gat_layers=model_config.num_gat_layers,
            num_heads=model_config.num_heads,
            gat_dropout=model_config.gat_dropout,
            num_lstm_layers=model_config.num_lstm_layers,
            lstm_dropout=model_config.lstm_dropout,
        ).to(self.device)

        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def calculate_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
    ) -> dict[str, float]:
        """Calculate multi-label classification metrics."""
        y_pred_binary = (y_pred > threshold).float()
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred_binary.cpu().numpy()

        # Component-wise metrics
        component_metrics = {}
        for i, component in enumerate(model_config.component_names):
            try:
                component_metrics[component] = {
                    "precision": precision_score(
                        y_true_np[:, i], y_pred_np[:, i], zero_division=0
                    ),
                    "recall": recall_score(
                        y_true_np[:, i], y_pred_np[:, i], zero_division=0
                    ),
                    "f1": f1_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0),
                    "accuracy": accuracy_score(y_true_np[:, i], y_pred_np[:, i]),
                }
            except Exception as e:
                logger.warning(f"Error calculating metrics for {component}: {e}")
                component_metrics[component] = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "accuracy": 0,
                }

        # Macro averages
        macro_precision = np.mean([m["precision"] for m in component_metrics.values()])
        macro_recall = np.mean([m["recall"] for m in component_metrics.values()])
        macro_f1 = np.mean([m["f1"] for m in component_metrics.values()])

        # Micro averages (global)
        micro_precision = precision_score(
            y_true_np, y_pred_np, average="micro", zero_division=0
        )
        micro_recall = recall_score(
            y_true_np, y_pred_np, average="micro", zero_division=0
        )
        micro_f1 = f1_score(y_true_np, y_pred_np, average="micro", zero_division=0)

        return {
            "component_metrics": component_metrics,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "accuracy": accuracy_score(y_true_np, y_pred_np),
        }

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

            # Reshape targets to match logits shape
            # logits: [batch_size, 7]
            # batch.y: [batch_size * 7] -> need to reshape to [batch_size, 7]
            batch_size = logits.size(0)
            targets = batch.y.view(batch_size, -1)  # ✅ ДОБАВЬ ЭТО

            loss = self.criterion(
                logits, targets
            )  # ✅ Используй targets вместо batch.y

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).detach())
            all_targets.append(targets.detach())  # ✅ Используй targets

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(train_loader)

        return metrics

    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

                # Reshape targets to match logits
                batch_size = logits.size(0)
                targets = batch.y.view(batch_size, -1)  # ✅ ДОБАВЬ ЭТО

                loss = self.criterion(logits, targets)  # ✅ Используй targets

                total_loss += loss.item()
                all_preds.append(torch.sigmoid(logits))
                all_targets.append(targets)  # ✅ Используй targets

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.calculate_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(val_loader)

        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "val_loss": val_loss,
            "config": {
                "model_config": model_config.__dict__,
                "training_config": self.config.__dict__,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Always save latest
        latest_path = (
            Path(self.config.model_save_path).parent / "gnn_classifier_latest.ckpt"
        )
        torch.save(checkpoint, latest_path)

        # Save best model
        if is_best:
            torch.save(checkpoint, self.config.model_save_path)
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

    def train(self, epochs: int = None):
        """Main training loop."""
        epochs = epochs or self.config.epochs

        # Setup data and model
        train_loader, val_loader, test_loader = self.setup_data()
        self.model = self.setup_model()

        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        logger.info(f"Starting training for {epochs} epochs...")

        no_improvement_count = 0

        for epoch in range(epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.train_metrics.append(train_metrics)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            self.val_metrics.append(val_metrics)

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Check for improvement
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics["loss"], is_best)

            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics['macro_f1']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping
            if no_improvement_count >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final evaluation
        logger.info("Training completed!")
        logger.info(
            f"Best epoch: {self.best_epoch} with val_loss: {self.best_val_loss:.4f}"
        )

        # Test evaluation
        test_metrics = self.validate_epoch(test_loader)
        logger.info(
            f"Test Metrics - Loss: {test_metrics['loss']:.4f}, "
            f"F1: {test_metrics['macro_f1']:.4f}"
        )

        # Save training history
        self.save_training_history()

    def save_training_history(self):
        """Save training metrics history."""
        history = {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }

        history_path = (
            Path(self.config.model_save_path).parent / "training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN hydraulic diagnostics model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_config.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=training_config.batch_size,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr", type=float, default=training_config.learning_rate, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=training_config.device,
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=training_config.data_path,
        help="Path to training data",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()

        # Update config
        training_config.epochs = args.epochs
        training_config.batch_size = args.batch_size
        training_config.learning_rate = args.lr
        training_config.device = args.device
        training_config.data_path = args.data_path

        # Create trainer and train
        trainer = HydraulicTrainer(training_config)
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
