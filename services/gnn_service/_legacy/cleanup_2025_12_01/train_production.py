# services/gnn_service/train_production.py
"""
Production-grade training pipeline с checkpointing, early stopping, logging.
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import logging
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import model_config, training_config
from model_dynamic_gnn import create_model
from data_loader_dynamic import create_dynamic_dataloaders
from schemas import EquipmentMetadata
from shared.observability.audit_logger import get_audit_logger, AuditEventType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping для предотвращения overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logger.info(f"Validation loss improved: {self.best_loss:.6f} -> {val_loss:.6f}")
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """
    Сохранение checkpoint с полным состоянием.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': model_config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    
    # Periodic checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.tar"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Best model checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.tar"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model updated: val_loss={val_loss:.6f}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, float]:
    """
    Загрузка checkpoint для resume training.
    """
    checkpoint = torch.load(checkpoint_path, map_location=model_config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    logger.info(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.6f}")
    return epoch, val_loss


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str
) -> float:
    """
    Training loop для одной эпохи.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        try:
            # Feature extraction
            feats = {k: v.to(device) for k, v in batch["component_features"].items()}
            y_true = batch["targets"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            health, degradation, attention = model(feats)
            
            # Multi-label BCE loss
            loss = loss_fn(health, y_true)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        except Exception as e:
            logger.error(f"Training batch error: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str
) -> float:
    """
    Validation loop.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                feats = {k: v.to(device) for k, v in batch["component_features"].items()}
                y_true = batch["targets"].to(device)
                
                health, degradation, attention = model(feats)
                loss = loss_fn(health, y_true)
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.error(f"Validation batch error: {e}")
                continue
    
    return total_loss / max(num_batches, 1)


def main():
    """
    Main training function.
    """
    logger.info("=" * 80)
    logger.info("Starting production training pipeline")
    logger.info("=" * 80)
    
    # Load metadata
    with open(training_config.metadata_path) as f:
        metadata = EquipmentMetadata(**json.load(f))
    
    logger.info(f"Loaded metadata: {len(metadata.components)} components")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dynamic_dataloaders(
        training_config.data_path,
        training_config.metadata_path,
        batch_size=training_config.batch_size,
        sequence_length=training_config.sequence_length,
        num_workers=training_config.num_workers
    )
    
    logger.info(
        f"Dataloaders created: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader)} batches"
    )
    
    # Create model
    device = model_config.device
    logger.info(f"Device: {device}")
    
    model = create_model(metadata, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {num_params:,} parameters")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # TensorBoard
    run_name = f"gnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    logger.info(f"TensorBoard logs: runs/{run_name}")
    
    # Checkpoint directory
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)
    
    for epoch in range(training_config.max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{training_config.max_epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        logger.info(f"Train Loss: {train_loss:.6f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, loss_fn, device)
        logger.info(f"Val Loss: {val_loss:.6f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        logger.info(f"Learning Rate: {current_lr:.6e}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % 5 == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_dir, is_best=is_best
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    
    logger.info("=" * 80)
    logger.info("Training completed")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Checkpoints saved in: {checkpoint_dir}")
    logger.info("=" * 80)
    
    # Audit log
    audit_logger = get_audit_logger()
    await audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_CONFIG_CHANGE,
        user_id="training_pipeline",
        tenant_id="system",
        resource="gnn_model",
        action="training_completed",
        metadata={
            "epochs": epoch + 1,
            "best_val_loss": best_val_loss,
            "model_params": num_params
        }
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
