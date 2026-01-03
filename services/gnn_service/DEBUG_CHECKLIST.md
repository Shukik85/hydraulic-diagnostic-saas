# Debug Checklist: Phase 2 Integration

**Purpose**: Verify Phase 2 integration fixes are working  
**Target**: Model output verification, inference compatibility, schema validation  
**Time**: ~30 minutes total

---

## Pre-Debug Setup

- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Model checkpoint available: `models/v2.1.0.ckpt`
- [ ] Database accessible (optional for basic tests)

---

## 1. Model Output Verification (5 min)

### Test: Model returns correct v2.1.0 structure

```bash
python << 'EOF'
from src.models import UniversalTemporalGNNv2, ModelConfig
import torch
from torch_geometric.data import Data

# Create model
config = ModelConfig()
model = UniversalTemporalGNNv2(config)
model.eval()

# Create sample data
data = Data(
    x=torch.randn(5, 34),
    edge_index=torch.tensor([[0,1,2,3,4], [1,2,3,4,0]], dtype=torch.long),
    edge_attr=torch.randn(5, 14)
)

# Forward pass
with torch.no_grad():
    outputs = model(data, temporal=False)

print("\u2705 Output type:", type(outputs).__name__)  # Should be: dict
print("\u2705 Top-level keys:", list(outputs.keys()))  # Should be: ['component', 'graph']
print("\u2705 Component keys:", list(outputs['component'].keys()))  # Should be: ['health', 'anomaly']
print("\u2705 Graph keys:", list(outputs['graph'].keys()))  # Should be: ['health', 'degradation', 'anomaly', 'rul']

# Verify shapes
assert outputs['component']['health'].shape == (5, 1), "Component health shape wrong"
assert outputs['component']['anomaly'].shape == (5, 9), "Component anomaly shape wrong"
assert outputs['graph']['health'].shape == (1, 1), "Graph health shape wrong"
assert outputs['graph']['rul'].shape == (1, 1), "Graph RUL shape wrong"

print("\n✅ ALL CHECKS PASSED")
EOF
```

**Expected Output**:
```
✅ Output type: dict
✅ Top-level keys: ['component', 'graph']
✅ Component keys: ['health', 'anomaly']
✅ Graph keys: ['health', 'degradation', 'anomaly', 'rul']
✅ ALL CHECKS PASSED
```

**If Failed**:
- [ ] Check model file: `src/models/universal_temporal_gnn.py`
- [ ] Verify forward() method returns nested dict
- [ ] Check output layer definitions

---

## 2. Inference Engine Compatibility (5 min)

### Test: Inference engine handles dict output

```bash
python << 'EOF'
from src.inference.inference_engine import InferenceEngine
from src.inference.model_manager import ModelManager
from torch_geometric.data import Data
import torch

print("Testing inference engine dict handling...\n")

# Check inference_engine.py code
with open('src/inference/inference_engine.py', 'r') as f:
    content = f.read()
    
    # Should NOT have this pattern (old):
    if 'health, degradation, anomaly = model' in content:
        print("\u274c ERROR: Found tuple unpacking (old pattern)")
        print("   Location: _inference_single() method")
        print("   Fix: Change to: outputs = model(...)")
    else:
        print("\u2705 No tuple unpacking found (good!)")
    
    # Should have this pattern (new):
    if "outputs = model" in content and "outputs_dict" not in content:
        print("\u2705 Dict unpacking found (correct!)")
    elif "outputs_dict" in content:
        print("\u2705 Dict structure handling found")
    else:
        print("\u26a0️ WARNING: Check _inference_single() implementation")

print("\n✅ CODE STRUCTURE CHECK COMPLETE")
EOF
```

**Expected Output**:
```
Testing inference engine dict handling...
✅ No tuple unpacking found (good!)
✅ Dict unpacking found (correct!)
✅ CODE STRUCTURE CHECK COMPLETE
```

**If Failed**:
- [ ] Open `src/inference/inference_engine.py`
- [ ] Find `_inference_single()` method
- [ ] Replace tuple unpacking with dict: `outputs = model(...)`
- [ ] Verify `_postprocess()` uses `outputs['component']` and `outputs['graph']`

---

## 3. Response Schema Validation (5 min)

### Test: Response includes all v2.1.0 fields

```bash
python << 'EOF'
from src.schemas.responses import PredictionResponse
import inspect

print("Checking PredictionResponse schema...\n")

# Get fields
sig = inspect.signature(PredictionResponse.__init__)
fields = [p for p in sig.parameters if p != 'self']

print(f"Fields found: {len(fields)}")
print(f"Field list: {fields}\n")

# Check for required Phase 2 fields
required_fields = ['rul_hours', 'component_predictions']
for field in required_fields:
    if field in fields:
        print(f"\u2705 Found: {field}")
    else:
        print(f"\u274c MISSING: {field}")

print("\n✅ SCHEMA VALIDATION COMPLETE")
EOF
```

**Expected Output**:
```
Checking PredictionResponse schema...

Fields found: X
Field list: [...]  # includes rul_hours, component_predictions

✅ Found: rul_hours
✅ Found: component_predictions
✅ SCHEMA VALIDATION COMPLETE
```

**If Failed**:
- [ ] Open `src/schemas/responses.py`
- [ ] Add `rul_hours: float` field
- [ ] Add `component_predictions: List[ComponentDiagnosis]` field
- [ ] Update docstring

---

## 4. Unit Tests (10 min)

### Test: All 21 model tests pass

```bash
pytest tests/test_universal_temporal_gnn.py -v
```

**Expected Output**:
```
test_valid_config PASSED
test_invalid_dropout PASSED
test_model_initialization PASSED
test_forward_single_graph PASSED
... [18 more tests]
test_gradient_flow PASSED

==================== 21 passed in 2.45s ====================
```

**If Any Test Fails**:
- [ ] Read error message carefully
- [ ] Check if error is about output format
- [ ] Verify model code returns nested dict
- [ ] Run single test: `pytest tests/test_universal_temporal_gnn.py::TestUniversalTemporalGNNv2::test_forward_single_graph -xvs`

---

## 5. Integration Test (10 min)

### Test: Full pipeline works

```bash
pytest tests/integration/ -v
```

**Expected Output**:
```
test_diagnose_endpoint PASSED
test_with_model_loading PASSED
test_full_pipeline PASSED

==================== X passed in Y.XXs ====================
```

**If Failed**:
- [ ] Check error is not about tuple unpacking
- [ ] Check response schema has rul_hours and component_predictions
- [ ] Verify inference_engine._postprocess() builds correct response
- [ ] Run with verbose: `pytest tests/integration/ -xvs`

---

## 6. Manual API Test (5 min)

### Test: Start service and call endpoint

```bash
# Terminal 1: Start service
uvicorn src.api.main:app --reload

# Terminal 2: Call endpoint
curl -X POST http://localhost:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "debug_test",
    "topology_id": "standard_pump_system",
    "timestamp": "2026-01-03T23:30:00Z",
    "sensor_readings": {
      "pump_main": {"pressure": 150.5, "temperature": 65.2},
      "valve_control": {"position": 0.75, "leakage": 0.01}
    }
  }' | jq
```

**Expected Response**:
```json
{
  "status": "success",
  "model_version": "v2.1.0",
  "equipment_id": "debug_test",
  "diagnosis": {
    "component_predictions": [...],
    "system_predictions": {
      "health": 0.85,
      "degradation_rate": 0.12,
      "anomalies": {...},
      "rul_hours": 248.5
    },
    "inference_time_ms": 42.3
  }
}
```

**Key Checks**:
- [ ] Status is "success" (not error)
- [ ] Model version is "v2.1.0"
- [ ] Response includes `rul_hours`
- [ ] Response includes `component_predictions`
- [ ] HTTP status is 200 (not 500)
- [ ] Inference time < 100ms

**If Failed**:

**Error: TypeError about unpacking**
```
TypeError: cannot unpack non-iterable dict object
```
- [ ] Model returns dict, inference expects tuple
- [ ] Fix: Update `_inference_single()` in inference_engine.py

**Error: Missing fields**
```
KeyError: 'rul_hours'
```
- [ ] Response schema incomplete
- [ ] Fix: Add fields to PredictionResponse

**Error: 500 Internal Server Error**
```
Internal Server Error
```
- [ ] Check logs: `tail -f logs/gnn_service.log`
- [ ] Look for exact error
- [ ] Fix accordingly

---

## 7. Performance Check (Optional - 5 min)

### Test: Latency and memory

```bash
python << 'EOF'
import time
import torch
from src.models import UniversalTemporalGNNv2, ModelConfig
from torch_geometric.data import Data

model = UniversalTemporalGNNv2(ModelConfig())
model.eval()

# Create sample data
data = Data(
    x=torch.randn(100, 34),
    edge_index=torch.randint(0, 100, (2, 200)),
    edge_attr=torch.randn(200, 14)
)

# Warm up
with torch.no_grad():
    _ = model(data, temporal=False)

# Timing
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = model(data, temporal=False)
latency = (time.time() - start) / 10 * 1000  # ms

print(f"\u2705 Average latency: {latency:.1f}ms")
print(f"Target: < 100ms")
if latency < 100:
    print("\u2705 PASS")
else:
    print("\u26a0️ WARNING: Latency higher than target")
EOF
```

**Expected**:
```
✅ Average latency: 45.2ms
Target: < 100ms
✅ PASS
```

---

## Common Issues & Fixes

### Issue 1: TypeError: cannot unpack non-iterable dict object

**Cause**: Model returns dict, inference tries tuple unpacking  
**Location**: `src/inference/inference_engine.py` line ~320  
**Fix**:
```python
# BEFORE:
health, degradation, anomaly = model(...)

# AFTER:
outputs = model(...)
```

### Issue 2: KeyError: 'rul_hours'

**Cause**: Response schema missing field  
**Location**: `src/schemas/responses.py`  
**Fix**: Add field to PredictionResponse

### Issue 3: Test fails with shape mismatch

**Cause**: Model or inference returning wrong shape  
**Fix**:
- [ ] Check tensor shapes in model
- [ ] Verify batch dimension handling
- [ ] Check squeeze() operations

### Issue 4: 500 error on API call

**Cause**: Check logs for exact error  
**Fix**:
```bash
# View logs
tail -f logs/gnn_service.log

# Or run with debug
uvicorn src.api.main:app --reload --log-level debug
```

---

## Checklist Completion

- [ ] Step 1: Model output verification PASSED
- [ ] Step 2: Inference engine compatibility PASSED
- [ ] Step 3: Response schema validation PASSED
- [ ] Step 4: Unit tests (21/21) PASSED
- [ ] Step 5: Integration tests PASSED
- [ ] Step 6: Manual API test PASSED
- [ ] Step 7: Performance check PASSED (optional)

**If all PASSED**: Phase 2 integration is complete ✅

**If any FAILED**: Use "Common Issues" section to debug

---

## Support

**Questions about tests?** Check test file: `tests/test_universal_temporal_gnn.py`  
**Questions about integration?** Check: `src/inference/inference_engine.py`  
**Questions about schema?** Check: `src/schemas/responses.py`  
**Questions about model?** Check: `src/models/universal_temporal_gnn.py`

---

**Last Updated**: January 3, 2026  
**Purpose**: Verify Phase 2 integration fixes  
**Expected Time**: ~30 minutes
