"""
Complete Enhanced Training Script for F1 > 90%

Integration of:
- Enhanced model (model_v2.py)
- Enhanced dataset with augmentation (dataset_v2.py)
- Label smoothing loss
- CosineAnnealing scheduler
- Gradient clipping
- Advanced logging

Usage:
    python train_v2.py
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from config import model_config
from dataset_v2 import create_data_loaders

# Import enhanced modules
from model_v2 import create_enhanced_model
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_v2.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with label smoothing."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="mean"
        )


class EnhancedTrainer:
    """Enhanced trainer for F1 > 90%."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip = gradient_clip

        # Loss
        self.criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )

        # Tracking
        self.best_val_f1 = 0.0
        self.history = {
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": [],
        }

        logger.info("=" * 70)
        logger.info("ENHANCED TRAINER INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"  Device: {device}")
        logger.info(f"  Label smoothing: {label_smoothing}")
        logger.info(f"  Gradient clipping: {gradient_clip}")
        logger.info(f"  Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
        logger.info("  Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
        logger.info("=" * 70)

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training", ncols=100)
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

            # Reshape targets
            batch_size = logits.size(0)
            targets = batch.y.view(batch_size, -1).float()

            # Loss
            loss = self.criterion(logits, targets)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            # Update progress
            if batch_idx % 100 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                    }
                )

        # Calculate epoch metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return {"loss": total_loss / len(self.train_loader), "f1": f1}

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validation", ncols=100):
            batch = batch.to(self.device)

            logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

            batch_size = logits.size(0)
            targets = batch.y.view(batch_size, -1).float()

            loss = self.criterion(logits, targets)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        return {
            "loss": total_loss / len(self.val_loader),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    @torch.no_grad()
    def test(self) -> dict:
        """Evaluate on test set."""
        logger.info("\n" + "=" * 70)
        logger.info("TESTING ON TEST SET")
        logger.info("=" * 70)

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in tqdm(self.test_loader, desc="Testing", ncols=100):
            batch = batch.to(self.device)

            logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

            batch_size = logits.size(0)
            targets = batch.y.view(batch_size, -1).float()

            loss = self.criterion(logits, targets)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        # Per-component metrics
        component_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

        results = {
            "loss": total_loss / len(self.test_loader),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "component_f1": component_f1.tolist(),
        }

        logger.info("\nTest Results:")
        logger.info(f"  Loss: {results['loss']:.4f}")
        logger.info(f"  F1: {results['f1']:.4f} ({results['f1'] * 100:.2f}%)")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info("\nPer-component F1:")
        for name, score in zip(
            model_config.component_names, component_f1, strict=False
        ):
            logger.info(f"  {name}: {score:.4f}")
        logger.info("=" * 70)

        return results

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint."""
        Path("models").mkdir(exist_ok=True)

        # Save model state (just the GNN, not wrapper)
        if hasattr(self.model, "gnn"):
            model_state = self.model.gnn.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_val_f1": self.best_val_f1,
        }

        # Save latest
        torch.save(checkpoint, "models/enhanced_model_latest.ckpt")

        # Save best
        if is_best:
            torch.save(checkpoint, "models/enhanced_model_best.ckpt")
            logger.info(
                f"New best model! Val F1: {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)"
            )

    def train(self, num_epochs: int = 100, patience: int = 15):
        """Train the model."""
        logger.info(
            f"\nStarting training for {num_epochs} epochs (patience={patience})"
        )
        logger.info("=" * 70 + "\n")

        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info("=" * 70)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update scheduler
            if hasattr(self, "scheduler"):
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(
                f"  Train Loss: {train_metrics['loss']:.4f} | Train F1: {train_metrics['f1']:.4f}"
            )
            logger.info(
                f"  Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} ({val_metrics['f1'] * 100:.2f}%)"
            )
            logger.info(
                f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}"
            )
            logger.info(f"  Learning Rate: {current_lr:.6f}")

            # Save history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["learning_rate"].append(current_lr)

            # Check improvement
            if val_metrics["f1"] > self.best_val_f1:
                improvement = val_metrics["f1"] - self.best_val_f1
                self.best_val_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                epochs_without_improvement = 0
                logger.info(f" ^ Improvement: +{improvement:.4f}")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f" -> No improvement ({epochs_without_improvement}/{patience})"
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"\nII  Early stopping at epoch {epoch + 1}")
                logger.info(
                    f"   Best Val F1: {self.best_val_f1:.4f} ({self.best_val_f1 * 100:.2f}%)"
                )
                break

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        # Save history
        with open("models/enhanced_training_history.json", "w") as f:  # noqa: PTH123
            json.dump(self.history, f, indent=2)

        # Final test
        test_results = self.test()

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 70)
        logger.info(
            f"Best Val F1: {self.best_val_f1:.4f} ({self.best_val_f1 * 100:.2f}%)"
        )
        logger.info(
            f"Test F1: {test_results['f1']:.4f} ({test_results['f1'] * 100:.2f}%)"
        )
        logger.info("=" * 70)


def main():
    """Main training function."""

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Data loaders (with augmentation)
    logger.info("\nCreating data loaders with augmentation...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=12, num_workers=0
    )

    # Model
    logger.info("\nCreating enhanced model...")
    model = create_enhanced_model(device)

    # Trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.1,
        gradient_clip=1.0,
    )

    # Train
    trainer.train(num_epochs=100, patience=15)


if __name__ == "__main__":
    main()
