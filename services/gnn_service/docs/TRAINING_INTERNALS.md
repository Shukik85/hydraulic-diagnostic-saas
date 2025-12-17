# PyTorch Lightning Training Internals & Backward Graph Analysis

## 🎯 Overview

When you call `trainer.fit(module, train_loader, val_loader)`, PyTorch Lightning executes a complex chain of operations. This guide explains what happens under the hood and where implicit `.backward()` calls can occur.

---

## 📊 Execution Flow: What Happens in `trainer.fit()`?

```
trainer.fit(module, train_loader, val_loader)
    ↓
call._call_and_handle_interrupt()
    ↓ (wraps with exception handling)
call._fit_impl()
    ↓
self._run()
    ↓
self._run_stage()
    ↓
self.fit_loop.run()
    ↓
for epoch in range(max_epochs):
    self.advance()
        ↓
    self.epoch_loop.run()
        ↓
    for batch_idx, batch in enumerate(train_loader):
        ← SANITY CHECKS (num_sanity_val_steps) happen HERE
        ← First validation pass on validation data
        
        self.automatic_optimization.run(optimizer, batch_idx, kwargs)
            ↓
        closure = train_step_and_backward_closure
            ↓
        training_step(batch)  # ← Your code, returns loss
            ↓
        loss.backward()       # ← IMPLICIT .backward() #1
            ↓
        optimizer.step()      # ← Updates parameters
            ↓
        optimizer.zero_grad() # ← Clears gradients
        
        ← After each batch, validation may run (check_val_every_n_epoch)
        
    ← END OF EPOCH
    
    ← Validation loop (only at epoch end now)
    for batch in val_loader:
        validation_step(batch)  # ← Should use torch.no_grad()
    
    ← Scheduler step
    scheduler.step(val_loss)
    
    ← CRITICAL: Cleanup happens here (on_epoch_end())
```

---

## 🔴 Common Sources of Double Backward Errors

### 1. **Sanity Checks Caching Graph**

**What happens:**
```
Running 2 sanity checks...
  validation_step(batch_1) → computes loss
  validation_step(batch_2) → computes loss
  ← Graph is still in memory
  
training_step(batch_1) → tries to backward on cached graph
  → RuntimeError: Trying to backward through the graph a second time
```

**Solution:**
```python
# Dev mode: Disable sanity checks
num_sanity_val_steps=0

# Wrap validation_step with torch.no_grad()
with torch.no_grad():
    outputs = model(...)
    loss = compute_loss(...)
```

---

### 2. **Validation Between Batches Caching Graph**

**What happens:**
```
Epoch 1:
  training_step(batch_1)
    ↓
  loss.backward()  ← Graph used
  optimizer.step()
  
  ← check_val_every_n_epoch triggers validation
  
  validation_step(batch_1)
    ↓
  loss computed on SAME forward pass
  ← Graph is re-used!
  
  training_step(batch_2)
    ↓
  loss.backward()  ← ERROR: Graph already freed!
```

**Solution:**
- Ensure `torch.no_grad()` wraps entire validation computation
- Don't log metrics from validation that depend on graph

---

### 3. **Scheduler Caching Graph State**

**What happens:**
```python
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
    # ← Stores loss values and optimizer state
)

scheduler.step(val_loss)  # ← Implicitly references old tensors
```

**Solution:**
```python
# Use interval='epoch' + on_epoch_end cleanup
def on_epoch_end(self):
    torch.cuda.empty_cache()  # Clear GPU memory
    torch.cuda.synchronize()  # Ensure all operations done
```

---

### 4. **Optimizer Closure Re-computing Loss**

**What happens:**
```python
# Some optimizers use closure (not Adam, but e.g., LBFGS)
optimizer.step(closure=closure)

# closure function:
def closure():
    optimizer.zero_grad()
    loss = model(batch)  # ← Re-compute loss
    loss.backward()      # ← First backward
    return loss

# Then Lightning calls:
loss.backward()  # ← Second backward on SAME graph
```

**Solution:**
- Use optimizers that don't require closure (Adam, SGD)
- If you need closure, set `self.automatic_optimization = False` and handle it manually

---

### 5. **Custom Autograd Functions (GradientReversal, etc.)**

**What happens:**
```python
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x  # ← Saves context for backward

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# If GradientReversal is applied in loss that's used twice:
loss1 = model(batch)  # ← GradientReversal creates forward node
loss1.backward()      # ← First backward, frees context

loss2 = model(batch)  # ← Same graph structure
loss2.backward()      # ← Second backward, context already freed!
```

**Solution:**
- Ensure custom functions use `create_graph=False` if not needed for second-order gradients
- Use `retain_graph=False` (default) to free after first backward

---

## 🔧 Debugging Techniques

### Technique 1: Enable Anomaly Detection

```python
import torch

def main():
    # MOST POWERFUL: Shows full stack trace when error occurs
    torch.autograd.set_detect_anomaly(True)
    
    # ... rest of code ...
    trainer.fit(module, train_loader, val_loader)
```

**Output when error occurs:**
```
RuntimeError: Trying to backward through the graph a second time
(See above for full traceback)

Tensor-related functions were invoked via Python's C API to construct
variables that do not support automatic differentiation. Specify
requires_grad_() or a dtype before the call to C function.
```

---

### Technique 2: Log Loss Tensor Properties

```python
def training_step(self, batch, batch_idx):
    outputs = self(...)
    total_loss, loss_dict = self.compute_loss(outputs, batch)
    
    # DEBUG: Print loss properties
    logger.info(
        f"🔧 [Train] Loss properties:\n"
        f"   requires_grad: {total_loss.requires_grad}\n"
        f"   is_leaf: {total_loss.is_leaf}\n"
        f"   grad_fn: {total_loss.grad_fn}\n"
        f"   shape: {total_loss.shape}\n"
        f"   dtype: {total_loss.dtype}"
    )
    
    return total_loss

def validation_step(self, batch, batch_idx):
    with torch.no_grad():
        outputs = self(...)
        total_loss, loss_dict = self.compute_loss(outputs, batch)
        
        # DEBUG: Print loss properties
        logger.info(
            f"🧪 [Val] Loss properties:\n"
            f"   requires_grad: {total_loss.requires_grad}\n"
            f"   is_leaf: {total_loss.is_leaf}\n"
            f"   grad_fn: {total_loss.grad_fn}\n"
            f"   shape: {total_loss.shape}"
        )
    
    return total_loss
```

**Expected output:**
```
🔧 [Train] Loss properties:
   requires_grad: True
   is_leaf: False
   grad_fn: AddBackward0
   shape: torch.Size([])
   dtype: torch.float32

🧪 [Val] Loss properties:
   requires_grad: False  # ← Should be False due to torch.no_grad()
   is_leaf: False
   grad_fn: None        # ← Should be None due to torch.no_grad()
```

If validation shows `requires_grad=True` → validation is building graph!

---

### Technique 3: Trace Backward Calls

```python
import torch
import traceback

_backward_calls = []
_original_backward = torch.Tensor.backward

def traced_backward(self, *args, **kwargs):
    """Trace all backward calls with stack trace."""
    stack = traceback.format_stack()
    _backward_calls.append({
        'grad_fn': str(self.grad_fn),
        'shape': self.shape,
        'stack': '\n'.join(stack[-5:])  # Last 5 frames
    })
    
    if len(_backward_calls) > 1 and _backward_calls[-1]['grad_fn'] == _backward_calls[-2]['grad_fn']:
        print("💥 POSSIBLE DOUBLE BACKWARD:")
        print(f"First call stack:\n{_backward_calls[-2]['stack']}")
        print(f"Second call stack:\n{_backward_calls[-1]['stack']}")
    
    return _original_backward(self, *args, **kwargs)

torch.Tensor.backward = traced_backward
```

---

### Technique 4: Check Autograd Graph Lifecycle

```python
def check_graph_health(loss, label=""):
    """Inspect computation graph health."""
    logger.info(f"📊 Graph Health Check [{label}]")
    logger.info(f"   requires_grad: {loss.requires_grad}")
    logger.info(f"   is_leaf: {loss.is_leaf}")
    
    if loss.grad_fn is not None:
        fn = loss.grad_fn
        logger.info(f"   grad_fn type: {type(fn).__name__}")
        logger.info(f"   grad_fn: {fn}")
        
        # Walk the graph
        next_functions = fn.next_functions
        logger.info(f"   next_functions: {len(next_functions)} nodes")
        for i, (fn, idx) in enumerate(next_functions):
            if fn is not None:
                logger.info(f"      [{i}] {type(fn).__name__}")
```

---

## ✅ Best Practices Summary

| Aspect | Do ✅ | Don't ❌ |
|--------|------|----------|
| **Validation** | Wrap with `torch.no_grad()` | Return tensors with `requires_grad=True` |
| **Sanity Checks** | `num_sanity_val_steps=0` in dev mode | Leave default value in dev mode |
| **Scheduler** | Use `interval='epoch'` | Use `interval='step'` with multiple batches |
| **Cleanup** | Call `torch.cuda.empty_cache()` in `on_epoch_end()` | Let memory accumulate |
| **Logging** | Log detached tensors `.detach()` | Log tensors with gradients |
| **Precision** | Use `precision='32'` for stability | Use `precision='16'` (AMP caches graphs) |
| **Loss Function** | Always return scalar | Return non-scalar or maintain gradients |
| **Optimizer** | Use Adam/SGD (no closure) | Use LBFGS with manual backward |

---

## 🚀 Production Training Configuration

```yaml
# configs/training_temporal.yaml

training:
  max_epochs: 200
  num_workers: 11        # ← Speed up data loading
  devices: 1
  accelerator: gpu
  precision: "32"       # ← Prevent AMP caching
  
validation:
  check_interval: 1.0   # After each epoch
  num_sanity_val_steps: 2  # Production: 2, Dev: 0
```

---

## 🔥 Current Fix Status

✅ **Fixed in HydraulicGNNModule:**
- ✅ `validation_step()` wrapped with `torch.no_grad()`
- ✅ `test_step()` wrapped with `torch.no_grad()`
- ✅ `on_epoch_end()` clears GPU cache
- ✅ Scheduler uses `interval='epoch'`
- ✅ No manual `.backward()` calls
- ✅ No `retain_graph=True` anywhere
- ✅ Losses use `reduction='mean'` (scalars)
- ✅ `num_sanity_val_steps=0` in dev mode

✅ **Fixed in trainer.py:**
- ✅ Dev mode: `num_sanity_val_steps=0`
- ✅ Production: `num_sanity_val_steps=2`
- ✅ Proper trainer configuration

✅ **Fixed in losses.py:**
- ✅ `UncertaintyWeighting` uses buffers (no gradients)
- ✅ Weights detached with `.detach()`
- ✅ No learnable parameters interfering

---

## 📈 Performance Tips

### Enable Persistent Workers
```yaml
hardware:
  num_workers: 11
  pin_memory: true
  persistent_workers: true  # ← Reuse workers between epochs
```

### Use Gradient Accumulation
```yaml
training:
  accumulate_grad_batches: 2  # Simulate larger batch size
```

### Profile Backward Pass
```python
with torch.profiler.profile(...) as prof:
    trainer.fit(module, train_loader, val_loader)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

---

## 🧪 Testing Your Setup

```bash
# Quick sanity test (1 batch)
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode dev \
    --fast-dev-run

# Full dev training (10 epochs)
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode dev

# Production training (200 epochs)
python src/training/train_temporal.py \
    --config configs/training_temporal.yaml \
    --mode prod
```

---

## 📚 Further Reading

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch Lightning Training Loop](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [Automatic Differentiation Guide](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)

