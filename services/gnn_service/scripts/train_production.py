#!/usr/bin/env python
"""Production training script with auto data generation.

Full training pipeline:
1. Generate synthetic data (200 train + 50 val)
2. Train for 100 epochs with early stopping
3. Save best model checkpoint
4. Log to TensorBoard

Usage:
    python scripts/train_production.py

Expected runtime:
    - CPU: ~2-3 hours
    - GPU: ~15-30 minutes
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TemporalGraphDataset
from src.data.feature_config import FeatureConfig
from src.models import ModelConfig
from src.training.lightning_module import HydraulicGNNModule


def generate_synthetic_graph(graph_id: int, num_nodes: int = 10) -> Data:
    """Generate a synthetic graph (larger than smoke test).
    
    Args:
        graph_id: Graph ID (for reproducibility)
        num_nodes: Number of nodes (default 10 for more realistic)
        
    Returns:
        PyG Data object
    """
    torch.manual_seed(graph_id)
    
    num_edges = num_nodes * 2  # More edges for complex graphs
    
    # Node features: [num_nodes, 34]
    x = torch.randn(num_nodes, 34)
    
    # Edge index: Random connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    
    # Edge features: [num_edges, 14]
    edge_attr = torch.randn(num_edges, 14)
    
    # === Phase 2 Targets ===
    # Gradual degradation pattern
    health = max(0.3, 0.9 - (graph_id % 100) * 0.006)  # Slower degradation
    
    y_graph_health = torch.tensor([health], dtype=torch.float32)
    y_graph_degradation = torch.tensor([1.0 - health], dtype=torch.float32)
    y_graph_anomaly = torch.randint(0, 2, (9,), dtype=torch.float32)
    y_graph_rul = torch.tensor([health], dtype=torch.float32)  # Normalized
    
    # Component-level
    y_component_health = torch.rand(num_nodes) * 0.4 + 0.6  # [0.6, 1.0]
    y_component_anomaly = torch.randint(0, 2, (num_nodes, 9), dtype=torch.float32)
    
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_graph_health=y_graph_health,
        y_graph_degradation=y_graph_degradation,
        y_graph_anomaly=y_graph_anomaly,
        y_graph_rul=y_graph_rul,
        y_component_health=y_component_health,
        y_component_anomaly=y_component_anomaly,
        batch=batch,
    )


def main():
    """Run production training."""
    print("="*70)
    print("🚀 PRODUCTION TRAINING PIPELINE")
    print("="*70)
    
    start_time = time.time()
    
    # === 1. Configuration ===
    print("\n⚙️ Step 1: Configuration...")
    
    NUM_TRAIN = 200
    NUM_VAL = 50
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    NUM_NODES = 10  # Larger graphs
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        accelerator = "cpu"
        devices = 1
        print("⚠️ No GPU found, using CPU (will be slower)")
    
    print(f"\n📊 Training config:")
    print(f"   Train samples: {NUM_TRAIN}")
    print(f"   Val samples: {NUM_VAL}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Nodes per graph: {NUM_NODES}")
    print(f"   Accelerator: {accelerator}")
    
    # === 2. Generate data ===
    print(f"\n📊 Step 2: Generating {NUM_TRAIN + NUM_VAL} synthetic graphs...")
    data_start = time.time()
    
    train_graphs = [generate_synthetic_graph(i, NUM_NODES) for i in range(NUM_TRAIN)]
    val_graphs = [generate_synthetic_graph(i + NUM_TRAIN, NUM_NODES) for i in range(NUM_VAL)]
    
    data_time = time.time() - data_start
    print(f"✅ Generated in {data_time:.1f}s")
    print(f"   Graph size: {train_graphs[0].x.shape[0]} nodes, {train_graphs[0].edge_index.shape[1]} edges")
    print(f"   Features: {train_graphs[0].x.shape[1]}D nodes, {train_graphs[0].edge_attr.shape[1]}D edges")
    
    # === 3. Save to disk ===
    print("\n💾 Step 3: Saving data...")
    data_dir = Path("data/production_train")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"
    
    torch.save({'graphs': train_graphs}, train_path)
    torch.save({'graphs': val_graphs}, val_path)
    print(f"✅ Saved to {data_dir}")
    
    # === 4. Create DataLoaders ===
    print("\n🔄 Step 4: Creating DataLoaders...")
    feature_config = FeatureConfig(edge_in_dim=14)
    
    train_dataset = TemporalGraphDataset(
        data_path=train_path,
        feature_config=feature_config,
        split="train",
    )
    val_dataset = TemporalGraphDataset(
        data_path=val_path,
        feature_config=feature_config,
        split="val",
    )
    
    # More workers for faster loading
    num_workers = 4 if accelerator == "gpu" else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Workers: {num_workers}")
    
    # === 5. Create model ===
    print("\n🧠 Step 5: Initializing model...")
    model_config = ModelConfig(
        node_features=34,
        edge_features=14,
        gat_hidden_dim=128,  # Larger for production
        lstm_hidden_dim=256,
        gat_num_layers=3,  # Deeper
        lstm_num_layers=2,
        gat_num_heads=4,
        gat_dropout=0.2,
        lstm_dropout=0.2,
        head_dropout=0.1,
    )
    
    module = HydraulicGNNModule(
        model_config=model_config,
        learning_rate=0.001,  # Lower LR for stability
        scheduler_type="plateau",  # Reduce on plateau
        use_advanced_losses=False,  # Basic losses for simplicity
        use_confidence_weighting=False,
        use_domain_adversarial=False,
    )
    
    total_params = sum(p.numel() for p in module.parameters())
    print(f"✅ Model initialized")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # === 6. Setup trainer ===
    print("\n⚡ Step 6: Setting up trainer...")
    
    # Checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="production_training",
        version=f"run_{int(time.time())}",
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model-{epoch:02d}-{val/total_loss:.3f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,  # Save top 3 models
        save_last=True,  # Also save latest
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        patience=15,  # Stop if no improvement for 15 epochs
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        # NOTE: gradient_clip_val not supported with manual optimization
        # If needed, implement manual clipping in lightning_module.py:
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        precision="16-mixed" if accelerator == "gpu" else "32",  # Mixed precision for GPU
    )
    
    print(f"✅ Trainer configured")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Early stopping: patience={early_stop_callback.patience}")
    print(f"   Precision: {trainer.precision}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    # === 7. Train ===
    print("\n" + "="*70)
    print("🚀 Starting training...")
    print("="*70)
    print(f"\n📊 Monitor progress:")
    print(f"   TensorBoard: tensorboard --logdir lightning_logs/production_training")
    print(f"   URL: http://localhost:6006")
    print("\n" + "-"*70)
    
    training_start = time.time()
    
    try:
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        
        print("\n" + "-"*70)
        print("✅ Training completed successfully!")
        print("="*70)
        
        # === 8. Results ===
        print("\n📊 Training Results:")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print(f"   Training time: {training_time / 60:.1f} minutes")
        print(f"   Epochs completed: {trainer.current_epoch + 1}")
        print(f"   Best model: {checkpoint_callback.best_model_path}")
        print(f"   Best val loss: {checkpoint_callback.best_model_score:.4f}")
        
        # Load best checkpoint and show metrics
        if checkpoint_callback.best_model_path:
            ckpt = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
            print(f"\n💾 Best Checkpoint Details:")
            print(f"   Path: {checkpoint_callback.best_model_path}")
            print(f"   Epoch: {ckpt['epoch']}")
            print(f"   Global step: {ckpt['global_step']}")
            
        print("\n🎉 SUCCESS! Model trained and saved.")
        print("\n📋 Next steps:")
        print("   1. View metrics: tensorboard --logdir lightning_logs/production_training")
        print(f"   2. Load model: torch.load('{checkpoint_callback.best_model_path}')")
        print("   3. Run inference: python scripts/test_inference.py")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print(f"   Checkpoints saved in: {checkpoint_dir}")
        print(f"   Resume with: trainer.fit(module, ckpt_path='{checkpoint_dir}/last.ckpt')")
        return 1
        
    except Exception as e:
        print("\n\n❌ Training failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
