# Phase 2 Integration Fixes Guide

**Time**: 2-3 hours total  
**Complexity**: Medium (straightforward code changes)  
**Risk**: Low (well-understood fixes)

---

## Fix 1: inference_engine.py Output Format (2-3 hours)

### Location

File: `src/inference/inference_engine.py`  
Method: `_inference_single()`  
Line: ~320

### Current Code (BROKEN)

```python
def _inference_single(self, graph: Data, model_version: str) -> tuple:
    """Single graph inference."""
    with torch.inference_mode():
        # Load model
        model = self.model_registry.get_model(model_version)
        
        # Model expects: Data object
        # Inference TRIES to unpack as tuple (WRONG!)
        health, degradation, anomaly = model(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=batch,
        )
        # This line CRASHES with:
        # TypeError: cannot unpack non-iterable dict object
    
    return health, degradation, anomaly
```

### Problem Analysis

**Model v2.1.0 returns** (CORRECT):
```python
{
    'component': {'health': [...], 'anomaly': [...]},
    'graph': {'health': [...], 'degradation': [...], 'anomaly': [...], 'rul': [...]}
}
```

**Code tries to unpack** (WRONG):
```python
health, degradation, anomaly = {...}  # Dict has only 2 keys!
# CRASH!
```

### Fixed Code

```python
def _inference_single(self, graph: Data, model_version: str) -> dict:
    """Single graph inference.
    
    Returns:
        dict: Nested output structure from v2.1.0:
            {
                'component': {'health': Tensor, 'anomaly': Tensor},
                'graph': {'health': Tensor, 'degradation': Tensor, 
                         'anomaly': Tensor, 'rul': Tensor}
            }
    """
    with torch.inference_mode():
        # Load model
        model = self.model_registry.get_model(model_version)
        
        # Model returns dict with nested structure (v2.1.0)
        outputs = model(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=batch,
        )
        # outputs is now:
        # {'component': {...}, 'graph': {...}}
    
    return outputs
```

### Changes Made

1. **Return type**: `tuple` → `dict`
2. **Remove unpacking**: Delete tuple unpacking line
3. **Assign to variable**: `outputs = model(...)`
4. **Return dict**: `return outputs`
5. **Update docstring**: Explain new return structure

---

## Fix 2: _postprocess() Method Update (1-2 hours)

### Current Code (INCOMPLETE)

```python
def _postprocess(
    self, 
    equipment_id: str, 
    health: torch.Tensor,  # ← Only 3 parameters
    degradation: torch.Tensor,
    anomaly: torch.Tensor,
    inference_time: float
) -> PredictionResponse:
    """Post-process inference outputs."""
    
    # Extract scalars from tensors
    health_score = float(health.squeeze().cpu().item())
    degradation_rate = float(degradation.squeeze().cpu().item())
    anomaly_logits = anomaly.squeeze().cpu().numpy()
    
    # Build response (MISSING component-level and RUL)
    return PredictionResponse(
        equipment_id=equipment_id,
        health=HealthPrediction(score=health_score),
        degradation=DegradationPrediction(rate=degradation_rate),
        anomaly=AnomalyPrediction(predictions=anomaly_predictions),
        inference_time_ms=inference_time * 1000,
        # MISSING: rul_hours
        # MISSING: component_predictions
    )
```

### Problem Analysis

**Current issues**:
- Only 3 parameters (health, degradation, anomaly)
- Model returns 4 graph tasks + 2 component tasks = 6 total
- Missing RUL in response
- Missing component-level predictions
- Cannot use Phase 2 features

### Fixed Code

```python
def _postprocess(
    self, 
    equipment_id: str, 
    outputs_dict: dict,  # ← New: dict structure
    batch_indices: List[int],  # ← For component mapping
    inference_time: float
) -> PredictionResponse:
    """Post-process Phase 2 inference outputs.
    
    Args:
        equipment_id: Equipment identifier
        outputs_dict: Model outputs (nested dict from v2.1.0)
        batch_indices: Component indices for mapping
        inference_time: Inference duration (seconds)
    
    Returns:
        PredictionResponse: Complete v2.1.0 response with all 6 tasks
    """
    
    # Extract from nested structure
    component_outputs = outputs_dict['component']
    graph_outputs = outputs_dict['graph']
    
    # ==========================================
    # COMPONENT-LEVEL PREDICTIONS (v2.1.0)
    # ==========================================
    component_health_tensor = component_outputs['health']  # [N, 1]
    component_anomaly_tensor = component_outputs['anomaly']  # [N, 9]
    
    component_predictions = []
    for idx, component_id in enumerate(batch_indices):
        health_val = float(component_health_tensor[idx].cpu().item())
        anomaly_logits = component_anomaly_tensor[idx].cpu().numpy()  # [9]
        
        # Map anomaly indices to names
        anomaly_dict = {
            'normal': float(anomaly_logits[0]),
            'pressure_drop': float(anomaly_logits[1]),
            'overheating': float(anomaly_logits[2]),
            'cavitation': float(anomaly_logits[3]),
            'leakage': float(anomaly_logits[4]),
            'contamination': float(anomaly_logits[5]),
            'seal_wear': float(anomaly_logits[6]),
            'bearing_fault': float(anomaly_logits[7]),
            'valve_stuck': float(anomaly_logits[8]),
        }
        
        component_predictions.append(
            ComponentDiagnosis(
                component_id=str(component_id),
                health=health_val,
                anomalies=anomaly_dict
            )
        )
    
    # ==========================================
    # GRAPH-LEVEL PREDICTIONS (v2.1.0)
    # ==========================================
    graph_health = float(graph_outputs['health'].squeeze().cpu().item())
    degradation_rate = float(graph_outputs['degradation'].squeeze().cpu().item())
    
    graph_anomaly_tensor = graph_outputs['anomaly'].squeeze().cpu().numpy()  # [9]
    graph_anomaly_dict = {
        'normal': float(graph_anomaly_tensor[0]),
        'pressure_drop': float(graph_anomaly_tensor[1]),
        'overheating': float(graph_anomaly_tensor[2]),
        'cavitation': float(graph_anomaly_tensor[3]),
        'leakage': float(graph_anomaly_tensor[4]),
        'contamination': float(graph_anomaly_tensor[5]),
        'seal_wear': float(graph_anomaly_tensor[6]),
        'bearing_fault': float(graph_anomaly_tensor[7]),
        'valve_stuck': float(graph_anomaly_tensor[8]),
    }
    
    # RUL (NEW in Phase 2)
    rul_hours = float(graph_outputs['rul'].squeeze().cpu().item())
    
    # ==========================================
    # BUILD COMPLETE v2.1.0 RESPONSE
    # ==========================================
    return PredictionResponse(
        equipment_id=equipment_id,
        model_version='v2.1.0',
        timestamp=datetime.utcnow().isoformat(),
        
        # Component-level (NEW)
        component_predictions=component_predictions,
        
        # Graph-level
        health=HealthPrediction(score=graph_health),
        degradation=DegradationPrediction(rate=degradation_rate),
        anomaly=AnomalyPrediction(predictions=graph_anomaly_dict),
        
        # RUL (NEW)
        rul_hours=rul_hours,
        
        # Metadata
        inference_time_ms=inference_time * 1000,
    )
```

### Changes Made

1. **Parameter**: `health, degradation, anomaly` → `outputs_dict`
2. **Extract nested structures**: Component and graph outputs
3. **Process component predictions**: Loop through components
4. **Extract graph predictions**: All 4 graph tasks
5. **Extract RUL**: New field from v2.1.0
6. **Build complete response**: Include all 6 tasks
7. **Update docstring**: Document v2.1.0 structure

---

## Fix 3: Update Response Schema (1 hour)

### File: `src/schemas/responses.py`

### Add Component Diagnosis Class

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class ComponentDiagnosis:
    """Per-component diagnosis (v2.1.0)."""
    component_id: str
    health: float  # [0, 1]
    anomalies: Dict[str, float]  # 9 anomaly types
```

### Update PredictionResponse

```python
@dataclass
class PredictionResponse:
    """Complete v2.1.0 prediction response."""
    
    # Basic fields
    equipment_id: str
    model_version: str = 'v2.1.0'
    timestamp: datetime = field(default_factory=datetime.utcnow)
    inference_time_ms: float
    
    # Graph-level predictions (4 tasks)
    health: HealthPrediction
    degradation: DegradationPrediction
    anomaly: AnomalyPrediction
    
    # Component-level predictions (2 tasks) - NEW in v2.1.0
    component_predictions: List[ComponentDiagnosis] = field(default_factory=list)
    
    # RUL - NEW in v2.1.0
    rul_hours: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': 'success',
            'model_version': self.model_version,
            'equipment_id': self.equipment_id,
            'timestamp': self.timestamp.isoformat(),
            'diagnosis': {
                'component_predictions': [
                    {
                        'component_id': pred.component_id,
                        'health': pred.health,
                        'anomalies': pred.anomalies
                    }
                    for pred in self.component_predictions
                ],
                'system_predictions': {
                    'health': self.health.score,
                    'degradation_rate': self.degradation.rate,
                    'anomalies': self.anomaly.predictions,
                    'rul_hours': self.rul_hours,
                },
                'inference_time_ms': self.inference_time_ms,
            }
        }
```

---

## Integration Verification

### Step 1: Verify Model Output

```bash
python << 'EOF'
from src.models import UniversalTemporalGNNv2, ModelConfig
import torch
from torch_geometric.data import Data

model = UniversalTemporalGNNv2(ModelConfig())
model.eval()

data = Data(
    x=torch.randn(5, 34),
    edge_index=torch.tensor([[0,1,2,3,4], [1,2,3,4,0]], dtype=torch.long),
    edge_attr=torch.randn(5, 14)
)

with torch.no_grad():
    outputs = model(data, temporal=False)

assert isinstance(outputs, dict), "Model should return dict"
assert 'component' in outputs and 'graph' in outputs
assert outputs['component']['health'].shape == (5, 1)
assert outputs['graph']['rul'].shape == (1, 1)

print("✅ Model output correct (v2.1.0)")
EOF
```

### Step 2: Verify Inference Engine

```bash
# Check no tuple unpacking
grep -n "health, degradation, anomaly =" src/inference/inference_engine.py
# Should return: (empty)

# Check dict handling
grep -n "outputs = model" src/inference/inference_engine.py
# Should return: line number where outputs = model(...)
```

### Step 3: Run Tests

```bash
pytest tests/test_universal_temporal_gnn.py::TestUniversalTemporalGNNv2::test_forward_single_graph -xvs
# Should PASS

pytest tests/test_universal_temporal_gnn.py -v
# All 21 should PASS
```

### Step 4: Integration Test

```bash
pytest tests/integration/test_full_pipeline.py -v
# Should PASS
```

---

## Rollback Plan

If something breaks:

```bash
# Revert to last working commit
git reset --hard HEAD~1

# Or keep current and fix:
git diff src/inference/inference_engine.py  # Check changes
git checkout -- src/inference/inference_engine.py  # Revert file
```

---

## Common Pitfalls

1. **Forgetting to update _postprocess()** → Response will still be incomplete
2. **Not extracting RUL** → Phase 2 feature missing
3. **Not handling component predictions** → Cannot use component-level output
4. **Forgetting to update response schema** → New fields won't serialize

---

## Success Criteria

- [ ] `_inference_single()` returns dict (not tuple)
- [ ] `_postprocess()` handles nested dict structure
- [ ] Component predictions included in response
- [ ] RUL included in response
- [ ] All 21 unit tests pass
- [ ] Integration tests pass
- [ ] Manual API test returns v2.1.0 response
- [ ] No TypeErrors in logs

---

**Time**: ~2-3 hours total  
**Difficulty**: Medium  
**Impact**: Enables Phase 2 features (component-level diagnostics + RUL)
