"""Training script for GNN with SSL pretraining + supervised fine-tuning."""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Optional
import os

from .config import config
from .dataset import HydraulicGraphDataset
from .model import GNNClassifier
from .ssl_pretraining import SSLPretrainer


class SSLLightningModule(pl.LightningModule):
    """PyTorch Lightning module for SSL pretraining."""
    
    def __init__(self, model: SSLPretrainer, lr: float = config.ssl_lr):
        super().__init__()
        self.model = model
        self.lr = lr
    
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        loss, metrics = self.model.compute_ssl_loss(batch)
        
        for key, value in metrics.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.ssl_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


class SupervisedLightningModule(pl.LightningModule):
    """PyTorch Lightning module for supervised fine-tuning."""
    
    def __init__(self, model: GNNClassifier, lr: float = config.finetune_lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])
    
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        logits, _, _ = self.model(batch)
        loss = F.cross_entropy(logits, batch.y)
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch.y).float().mean()
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, _, _ = self.model(batch)
        loss = F.cross_entropy(logits, batch.y)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch.y).float().mean()
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", acc, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss, "val_accuracy": acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


def train_ssl(
    dataset: HydraulicGraphDataset,
    save_path: str = "./models/ssl_pretrained.ckpt",
) -> SSLPretrainer:
    """Run SSL pretraining.
    
    Args:
        dataset: Training dataset
        save_path: Path to save pretrained model
    
    Returns:
        Pretrained model
    """
    print("[1/2] Starting SSL Pretraining...")
    
    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    # Model
    ssl_model = SSLPretrainer()
    lightning_module = SSLLightningModule(ssl_model)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(save_path),
        filename="ssl_pretrained",
        monitor="ssl/total_loss",
        mode="min",
        save_top_k=1,
    )
    
    # Logger
    logger = TensorBoardLogger(config.log_dir, name="ssl_pretraining")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.ssl_epochs,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
    )
    
    # Train
    trainer.fit(lightning_module, train_loader)
    
    print(f"SSL Pretraining complete! Model saved to {save_path}")
    
    return ssl_model


def train_supervised(
    dataset: HydraulicGraphDataset,
    val_dataset: Optional[HydraulicGraphDataset] = None,
    ssl_model: Optional[SSLPretrainer] = None,
    save_path: str = "./models/gnn_classifier.ckpt",
) -> GNNClassifier:
    """Run supervised fine-tuning.
    
    Args:
        dataset: Training dataset
        val_dataset: Validation dataset (optional)
        ssl_model: Pretrained SSL model (if available)
        save_path: Path to save fine-tuned model
    
    Returns:
        Fine-tuned classifier
    """
    print("[2/2] Starting Supervised Fine-Tuning...")
    
    # DataLoaders
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    
    # Model
    classifier = GNNClassifier()
    
    # Transfer weights from SSL encoder
    if ssl_model is not None:
        print("Transferring weights from SSL pretrained encoder...")
        classifier.encoder.load_state_dict(ssl_model.encoder.state_dict())
    
    lightning_module = SupervisedLightningModule(classifier)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(save_path),
        filename="gnn_classifier_best",
        monitor="val/accuracy" if val_loader else "train/accuracy",
        mode="max",
        save_top_k=1,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/loss" if val_loader else "train/loss",
        patience=10,
        mode="min",
    )
    
    # Logger
    logger = TensorBoardLogger(config.log_dir, name="supervised_finetuning")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.finetune_epochs,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=1.0,
    )
    
    # Train
    trainer.fit(lightning_module, train_loader, val_loader)
    
    print(f"Supervised Fine-Tuning complete! Model saved to {save_path}")
    
    return classifier


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipment-ids", nargs="+", required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    
    # Load dataset
    full_dataset = HydraulicGraphDataset(
        root="./data",
        equipment_ids=args.equipment_ids,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    # Train/val split
    train_size = int(len(full_dataset) * (1 - args.val_split))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # SSL Pretraining
    ssl_model = train_ssl(train_dataset)
    
    # Supervised Fine-tuning
    classifier = train_supervised(train_dataset, val_dataset, ssl_model)
    
    print("✅ Training complete!")
