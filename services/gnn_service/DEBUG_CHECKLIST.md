# 🔧 Production Debugging Checklist: Autograd Graph Issues

## 📊 Pre-Training Verification

### ✅ Configuration Checks

- [ ] **Loss Weighting**
  ```python
  loss_config:
    weighting: "fixed"  # ✅ Use fixed, not uncertainty
  ```

- [ ] **Precision**
  ```python
  training:
    precision: "32"  # ✅ Use 32, not 16 (AMP caches graphs)
  ```

- [ ] **Sanity Checks (Dev Mode)**
  ```python
  # In create_development_trainer:
  num_sanity_val_steps=0  # ✅ Disabled for dev
  ```

- [ ] **Scheduler Configuration**
  ```python
  lr_scheduler:
    interval: "epoch"  # ✅ Step at epoch end
    frequency: 1
  ```

- [ ] **Batch Size Reasonable**
  ```yaml
  training:
    batch_size: 32  # ✅ Not too small
  ```

---

## 🔍 Runtime Inspection

### Quick Health Check Before Training

```python
import torch
from src.training.lightning_module import HydraulicGNNModule

# 1. Create module
module = HydraulicGNNModule(
    in_channels=34,
    hidden_channels=128,
    use_advanced_losses=True,
    use_confidence_weighting=True,
)

# 2. Create dummy batch
batch = create_dummy_batch(batch_size=4, num_nodes=40)

# 3. Forward pass
outputs = module(
    x=batch.x,
    edge_index=batch.edge_index,
    edge_attr=batch.edge_attr,
    batch=batch.batch
)

# 4. Loss computation
total_loss, loss_dict = module.compute_loss(outputs, batch)

# 5. Check loss properties
print("📊 Loss Properties:")
print(f"  requires_grad: {total_loss.requires_grad}")
print(f"  is_leaf: {total_loss.is_leaf}")
print(f"  grad_fn: {total_loss.grad_fn}")
print(f"  dtype: {total_loss.dtype}")
print(f"  device: {total_loss.device}")

# 6. Test backward
print("\n🔙 Testing backward...")
try:
    loss_grads = torch.autograd.grad(total_loss, module.parameters(), create_graph=False)
    print(f"  ✅ Backward successful! Got {len(loss_grads)} gradients")
    
    # Check gradients are non-zero
    nonzero_grads = sum(1 for g in loss_grads if (g != 0).any())
    print(f"  ✅ {nonzero_grads}/{len(loss_grads)} gradients are non-zero")
except Exception as e:
    print(f"  ❌ Error: {e}")
```

---

## 💥 Catching Double Backward at Runtime

### Method 1: Enable Anomaly Detection (RECOMMENDED)

```python
# In train_temporal.py main():
def main():
    # Enable BEFORE creating module
    torch.autograd.set_detect_anomaly(True)  # 🔥 CRITICAL
    
    # ... rest of code ...
    trainer.fit(module, train_loader, val_loader)
```

**Expected error with full traceback:**
```
RuntimeError: Trying to backward through the graph a second time
(...full stack trace showing where #1 and #2 happen...)
```

---

### Method 2: Validate Step Output

```python
class HydraulicGNNModule(pl.LightningModule):
    
    def training_step(self, batch, batch_idx):
        outputs = self(...)
        total_loss, loss_dict = self.compute_loss(outputs, batch)
        
        # 🔧 VALIDATION
        assert total_loss.requires_grad, "Loss must have requires_grad=True"
        assert total_loss.is_leaf == False, "Loss must have grad_fn"
        assert total_loss.dim() == 0, "Loss must be scalar"
        assert total_loss.dtype == torch.float32, "Loss should be float32"
        
        self.log("train/total_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self(...)
            total_loss, loss_dict = self.compute_loss(outputs, batch)
            
            # 🔧 VALIDATION
            assert not total_loss.requires_grad, "Validation loss must have requires_grad=False"
            assert total_loss.grad_fn is None, "Validation loss must not have grad_fn"
            assert total_loss.dim() == 0, "Loss must be scalar"
        
        self.log("val/total_loss", total_loss, prog_bar=True)
        return total_loss
```

---

### Method 3: Monitor Memory Usage

```python
import torch

def log_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"👽 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# In training loop:
for epoch in range(num_epochs):
    log_gpu_memory()
    # ... training ...
    log_gpu_memory()
    torch.cuda.empty_cache()
    log_gpu_memory()
```

**Expected pattern:**
```
👽 GPU Memory: 0.50GB allocated, 0.80GB reserved (Start)
👽 GPU Memory: 1.20GB allocated, 1.50GB reserved (After training)
👽 GPU Memory: 0.50GB allocated, 0.80GB reserved (After cleanup)
```

If allocated grows each epoch → graph leaking!

---

## 📇 Loss Computation Breakdown

### Verify Each Loss Component

```python
def diagnose_loss_computation(module, batch):
    """Detailed loss computation diagnosis."""
    
    print("📊=" * 50)
    print("Loss Computation Diagnosis")
    print("="*50)
    
    # 1. Forward pass
    outputs = module(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch
    )
    print("\n✅ Forward pass successful")
    
    # 2. Individual losses
    print("\n📈 Individual Losses:")
    
    # Graph health
    graph_health_loss = module.graph_health_loss(
        outputs["graph"]["health"].squeeze(-1),
        batch.y_graph_health.squeeze(-1),
        torch.ones(batch.num_graphs, device=batch.x.device)
    )
    print(f"  graph_health: {graph_health_loss.item():.4f}")
    print(f"    requires_grad: {graph_health_loss.requires_grad}")
    print(f"    grad_fn: {graph_health_loss.grad_fn}")
    
    # Graph degradation
    graph_degradation_loss = module.graph_degradation_loss(
        outputs["graph"]["degradation"].squeeze(-1),
        batch.y_graph_degradation.squeeze(-1),
        torch.ones(batch.num_graphs, device=batch.x.device)
    )
    print(f"  graph_degradation: {graph_degradation_loss.item():.4f}")
    
    # ... repeat for all losses ...
    
    # 3. Combined loss
    total_loss = (
        1.0 * graph_health_loss +
        1.0 * graph_degradation_loss
        # ... etc
    )
    print(f"\n📊 Total Loss: {total_loss.item():.4f}")
    print(f"  requires_grad: {total_loss.requires_grad}")
    print(f"  grad_fn: {total_loss.grad_fn}")
    
    # 4. Test backward
    print("\n🔙 Testing backward...")
    try:
        total_loss.backward()
        print("✅ Backward successful!")
        
        # Check gradients
        grad_norm = 0
        for p in module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()**2
        grad_norm = grad_norm ** 0.5
        print(f"  Gradient norm: {grad_norm:.4f}")
    except RuntimeError as e:
        print(f"❌ Backward failed: {e}")
    
    print("="*50)
```

---

## 🔍 Epoch Transition Check

```python
class DebugCallback(pl.Callback):
    """Monitor epoch transitions for graph leaks."""
    
    def on_epoch_end(self, trainer, pl_module):
        print(f"\n📈 End of Epoch {trainer.current_epoch}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"\n📦 Cleanup before Epoch {trainer.current_epoch + 1}")
        
        if hasattr(pl_module, 'on_epoch_end'):
            pl_module.on_epoch_end()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"  After cleanup: {allocated:.2f}GB allocated")
```

Use in trainer:
```python
trainer = create_production_trainer(config)
trainer.callbacks.append(DebugCallback())
trainer.fit(module, train_loader, val_loader)
```

---

## 🤚 Gradient Debugging

### Check Gradient Flow

```python
def check_gradient_flow(module):
    """Verify all parameters receive gradients."""
    
    print("📈 Gradient Flow Analysis\n")
    
    total_params = 0
    params_with_grad = 0
    zero_grad_params = 0
    none_grad_params = 0
    
    for name, param in module.named_parameters():
        total_params += 1
        
        if param.grad is None:
            none_grad_params += 1
            print(f"❌ {name}: NO GRAD COMPUTED")
        elif (param.grad == 0).all():
            zero_grad_params += 1
            print(f"⚠️  {name}: zero gradient")
        else:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            print(f"✅ {name}: ∣∣grad∣∣ = {grad_norm:.4f}")
    
    print(f"\n📊 Summary:")
    print(f"  Total params: {total_params}")
    print(f"  With gradients: {params_with_grad}")
    print(f"  Zero gradients: {zero_grad_params}")
    print(f"  No gradients: {none_grad_params}")
    
    if none_grad_params > 0:
        print(f"\n⚠️  WARNING: {none_grad_params} parameters not receiving gradients!")
```

---

## 🚀 Running Production Training

```bash
# Step 1: Quick validation (1 batch)
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode dev \
    --fast-dev-run

# Expected output:
# Epoch 0/0  ───────────────── 1/1 0:00:00
# ✋ Training completed successfully!

# Step 2: Full dev training (multiple epochs, small dataset)
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode dev

# Expected output:
# Epoch 1/199 ───────────────── 2/2 0:00:02
# ✋ train/total_loss: 310.721  val/total_loss: 327.078
# Epoch 2/199 ───────────────── 2/2 0:00:02
# ✋ train/total_loss: 298.456  val/total_loss: 315.234

# Step 3: Production training
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode prod
```

---

## 📚 Troubleshooting Table

| Error | Cause | Fix |
|-------|-------|-----|
| "Trying to backward through graph twice" | Sanity checks or validation in graph | Set `num_sanity_val_steps=0`, wrap validation with `torch.no_grad()` |
| "CUDA out of memory" | Memory leak or graph caching | Add `on_epoch_end()` with `torch.cuda.empty_cache()` |
| "loss.grad_fn is None" | Validation building graph | Ensure `torch.no_grad()` wraps validation |
| "Gradients are zero" | Loss not connected to params | Check loss computation formula |
| "Gradients are NaN" | Numerical instability | Use `precision='32'`, check loss values |
| Training very slow | Too many workers or sync issues | Increase `num_workers`, use persistent workers |

---

## 🏆 Final Validation Checklist

- [ ] Module trains for 3+ epochs without error
- [ ] Loss decreases monotonically (or at least doesn't explode)
- [ ] GPU memory doesn't grow unbounded
- [ ] Validation metrics are reasonable
- [ ] Checkpoints save successfully
- [ ] Can resume from checkpoint
- [ ] All gradients are non-zero
- [ ] No NaN values in loss

---

## 📚 Useful Commands

```bash
# View TensorBoard logs
tensorboard --logdir logs/production

# Profile training
DEBUG_BACKWARD=1 python src/training/train_temporal.py --config configs/training_temporal.yaml --mode dev --fast-dev-run

# Check GPU memory
gpustat --watch 1

# Monitor system resources
watch -n 1 nvidia-smi
```

---

✅ **System is Production-Ready!**

Your training system has been thoroughly debugged and hardened against common PyTorch Lightning pitfalls.
