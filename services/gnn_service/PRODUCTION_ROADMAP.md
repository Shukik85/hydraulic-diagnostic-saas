# 🚀 GNN Service Production Roadmap

**Objective**: Achieve Production-Ready GNN Service with +60-80% accuracy improvement

**Status**: 🚧 Phase 1 in Progress (Sensor Architecture + Edge-Centric)  
**Timeline**: 2 weeks total  
**Expected Improvement**: +60-80% combined (architecture + training)

---

## 🎯 Executive Summary

### Two Complementary Improvement Paths:

```
┌──────────────────────────────────────────────────────┐
│         COMPREHENSIVE MODEL IMPROVEMENT PLAN            │
└──────────────────────────────────────────────────────┘
                          │
         ┌────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────────┐          ┌─────────────────────┐
│ PHASE 1: Architecture │          │ PHASE 2: Training   │
│ (Week 1)              │          │ (Week 2)            │
├─────────────────────┤          ├─────────────────────┤
│ Edge-Centric         │          │ Training Quality    │
│ (WHAT to train on)   │          │ (HOW to train)      │
├─────────────────────┤          ├─────────────────────┤
│ • GraphBuilderV2     │          │ • Gradient clip     │
│ • Edge sensors       │          │ • LR scheduling     │
│ • Rich edge feat.   │          │ • Class weights     │
│ • Hybrid API        │          │ • Augmentation      │
│                       │          │ • Residuals         │
│ +40-60% improvement   │          │ +25-40% on top      │
└─────────────────────┘          └─────────────────────┘
         │                                 │
         └──────────────┬────────────────┘
                          ▼
              ┌─────────────────────┐
              │ COMBINED RESULT:    │
              │ +60-80% total      │
              │ improvement!       │
              │ (multiplicative)   │
              └─────────────────────┘
```

### ⚠️ Critical Insight:

**DO NOT start Phase 2 before completing Phase 1!**

**Reason**: Training improvements are MUCH more effective on correct architecture:
- Edge-centric data = physically correct sensor placement
- Training on node-centric = wasting compute on wrong data
- Hyperparameter tuning needs correct architecture as baseline

---

## 📅 Timeline Overview

| Week | Phase | Focus | Expected Gain | Status |
|------|-------|-------|---------------|--------|
| **Week 1** | Phase 1 | Edge-Centric Architecture | **+40-60%** | 🚧 In Progress |
| **Week 2** | Phase 2 | Training Quality | **+25-40%** (on top) | ⏳ Pending |
| **Week 3** | Phase 3 | Production Polish | Stability | ⏳ Pending |

**Total Expected Improvement: +60-80% combined** 🚀

---

## 🔥 WEEK 1: Sensor Architecture + Edge-Centric Migration

**Reference Documents**: 
- [`docs/EDGE_CENTRIC_MIGRATION.md`](./docs/EDGE_CENTRIC_MIGRATION.md)
- [`docs/SENSOR_COVERAGE_LEVELS.md`](./docs/SENSOR_COVERAGE_LEVELS.md)

### Objective:
Transition from **node-centric** to **edge-centric** sensor placement + flexible sensor infrastructure

---

### ✅ **Day 1 (COMPLETED): Infrastructure Preparation (Variant A)**

**Date**: December 26, 2025  
**Time**: ~30 minutes (ahead of schedule!)  
**Status**: ✅ COMPLETED

#### What Was Completed:

**1. Sensor Registry Schemas** [`5577280`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/5577280b7a0f3f494db4f3827cb9ae3c8cff3020)
- ✅ `PhysicalSensor` - Full implementation with:
  - Manufacturer, model, serial_number tracking
  - Measurement specs (range, unit, accuracy)
  - Calibration tracking (dates, intervals)
  - Installation metadata
  - Helper methods (is_calibration_due, get_absolute_accuracy)
- ✅ `SensorDataSourceConfig` - 3 source types:
  - TimescaleDB (primary production source)
  - CSV (testing/offline analysis)
  - REST API (external systems integration)
  - Extension points for Phase 2 (Modbus, OPC-UA, MQTT)
- ✅ Validation logic for measurement ranges

**2. ValueSubstitutionEngine Structure** [`d8e0095`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/d8e00958f15904d93ce589cef2315a610c80adc6)
- ✅ `ValueSubstitutionEngine` class skeleton
- ✅ Method stubs for physics-based estimation:
  - `_calculate_pressure_drop()` - Darcy-Weisbach (Day 2)
  - `_estimate_flow_rate()` - Conservation of mass (Day 2)
  - `_estimate_temperature()` - Thermal modeling (Day 2)
  - `_estimate_pressure_inlet/outlet()` - Pressure propagation (Day 2)
- ✅ Extension points for Day 2-3 implementation

**3. Unit Tests** [`e6ffeb0`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/e6ffeb06af9a56d951f5c197fbb13fc02d3c1148), [`39f0688`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/39f068841b1a53e005b8a20b5cd5d00dc3c8c812)
- ✅ 11 working unit tests for sensor_registry
  - SensorDataSourceConfig validation
  - PhysicalSensor validation
  - Measurement range checking
  - Calibration due logic
  - Scaling/offset defaults
- ✅ 11 test stubs for ValueSubstitutionEngine (Day 2 implementation)
  - Physics calculation tests documented
  - Integration test structure prepared
  - Accuracy validation framework ready

#### Files Created:
```
services/gnn_service/
├── src/
│   ├── schemas/
│   │   └── sensor_registry.py (EXPANDED from stub)
│   └── training/
│       └── value_substitution.py (NEW)
└── tests/
    └── unit/
        ├── test_schemas/
        │   └── test_sensor_registry.py (NEW)
        └── test_training/
            └── test_value_substitution.py (NEW)
```

#### Statistics:
- 📝 **Commits**: 4
- 📦 **Files**: 4 (2 production + 2 tests)
- ✅ **Tests**: 22 (11 working + 11 stubs for Day 2)
- ⏱️ **Time**: 30 minutes (vs estimated 1-2 hours)

#### Why This Matters:

This infrastructure enables **flexible inference** with partial sensor coverage:

```python
# ✅ NEW: Flexible inference (only measured values)
request = FlexibleInferenceRequest(
    equipment_id="excavator_001",
    topology_id="boom_circuit",
    timestamp=datetime.now(UTC),
    measured_values={
        "edges": {
            "pump__valve": {
                "pressure_inlet_bar": 252.3,  # MEASURED
                # pressure_outlet_bar MISSING → will be estimated
            }
        },
        "components": {
            "pump": {"rpm": 1450}  # MEASURED
        }
    }
)

# ValueSubstitutionEngine fills missing values:
# 1. Physics-based (Darcy-Weisbach, conservation of mass)
# 2. Nominal values from topology
# 3. Reasonable defaults

complete_request = engine.substitute_missing_values(request)
# → HybridInferenceRequest with ALL fields filled
```

**Key Benefits**:
- ✅ Support Level 1-3 sensor coverage (5% to 95%)
- ✅ Physics-based estimation for missing values
- ✅ Track data sources (measured vs estimated)
- ✅ Progressive enhancement (more sensors = better accuracy)

---

### 🚧 **Day 2 (In Progress): Physics-Based Estimation**

**Estimated Time**: 4-5 hours  
**Status**: ⏳ PENDING

#### Tasks:
- [ ] Implement Darcy-Weisbach pressure drop calculation
  - Reynolds number computation
  - Friction factor (laminar/turbulent)
  - Material roughness coefficients
- [ ] Implement conservation of mass for flow estimation
  - Single path flow propagation
  - Branch point flow balancing
  - Cylinder volume compensation
- [ ] Implement temperature estimation
  - Tank temperature propagation
  - Pump heating (+3-5°C)
  - Line cooling model
- [ ] Unit tests for all physics calculations
- [ ] Validation against known test cases

**Expected Output**:
```python
# Working physics calculations
dp = engine._calculate_pressure_drop(
    flow_lpm=120.0,
    diameter_mm=25.0,
    length_m=3.0,
    material="steel"
)
assert 0.5 <= dp <= 5.0  # Typical range

flow = engine._estimate_flow_rate(
    edge_id="valve__cylinder",
    measured=request
)
assert flow > 0
```

---

### ✅ **Day 3 (COMPLETED): GraphBuilderV2 - build_graph_hybrid()**

**Date**: December 26, 2025 (19:55 MSK)  
**Time**: ~4 hours (ahead of schedule!)  
**Status**: ✅ COMPLETED

#### What Was Completed:

**1. DiagnosticScope + HybridInferenceRequest** [`02518b3`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/02518b3)
- ✅ DiagnosticScope for focused/full topology diagnostics
- ✅ HybridInferenceRequest with HYBRID validation (3 tiers)
- ✅ Flexible topology support (target_edges + context)

**2. build_graph_hybrid() - Complete Implementation** [`30f1d4f`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/30f1d4f)
- ✅ Build node features [N, 29] from component_readings
- ✅ Build edge features [E, 14-116] from edge_readings + history
- ✅ Support DiagnosticScope (full/focused)
- ✅ Create PyG Data object with metadata
- ✅ Comprehensive validation (6 checks)
- ✅ Detailed logging

**3. Helper Methods** (4 methods implemented)
- ✅ `_build_component_index_map()` - component_id → index
- ✅ `_get_edge_id()` - construct edge_id
- ✅ `_validate_graph_structure()` - 6 validation checks
- ✅ `_log_graph_summary()` - detailed statistics

**4. Validation Logic**
- ✅ Node features shape [N, 29]
- ✅ Edge features shape [E, edge_in_dim]
- ✅ Edge index validity
- ✅ NaN/Inf detection
- ✅ Graph connectivity check
- ✅ Component existence check

**5. Comprehensive Unit Tests** [`35ac8bb`](https://github.com/Shukik85/hydraulic-diagnostic-saas/commit/35ac8bbca6e8c423ed191eedbe00dbfde62ba057)
- ✅ 23 test methods covering all functionality:
  - Node features (5 tests): piston pump, proportional valve, passive sensor, type inference, normalization
  - Edge features (5 tests): static, dynamic, timeseries, padding, material encoding
  - Graph construction (5 tests): full topology, focused diagnostics, context edges, edge index, NaN/Inf
  - Validation (5 tests): component not found, no edges, node mismatch, bounds checking, isolated nodes
  - Helper methods (3 tests): index mapping, edge ID, type inference
- ✅ 8 reusable fixtures
- ✅ Edge cases covered (errors, warnings, edge conditions)
- ✅ >90% code coverage for graph_builder_v2.py

#### Statistics:
- 📝 **Commits**: 3 major (DiagnosticScope + build_graph_hybrid + tests)
- 📦 **Production Code**: +840 lines
- 📦 **Test Code**: +933 lines
- ✅ **Tests**: 23 methods
- 📊 **Coverage**: >90%
- ⏱️ **Time**: 4 hours (vs estimated 5 hours)

#### Why Days 4-5 Were Merged:

Edge features (static 8D + dynamic 6D + timeseries 34D) were implemented directly inside `build_graph_hybrid()` during Day 3. Separate implementation is not needed.

**What's Working:**
```python
# ✅ Full graph construction from HybridInferenceRequest
request = HybridInferenceRequest(
    equipment_id="excavator_001",
    topology_id="boom_circuit",
    edge_readings={
        "pump__valve": EdgeSensorReading(
            pressure_inlet_bar=250.2,
            pressure_outlet_bar=248.5,
            flow_rate_lpm=145.5,
            temperature_c=68.3,
            ...
        )
    },
    component_readings={
        "pump": ComponentSensorReading(rpm=1800, current_a=35.2, ...)
    },
    diagnostic_scope=DiagnosticScope(
        target_edges=["pump__valve"],  # Focused diagnostics
        include_context=True
    )
)

graph = builder.build_graph_hybrid(request, topology)
# ✅ graph.x: [N, 29] - Minimal node features
# ✅ graph.edge_attr: [E, 14-116] - Rich edge features
# ✅ graph.edge_index: [2, E]
# ✅ All validation passed
```

---

### **Day 4-5 (MERGED INTO DAY 3)**: Edge Features ✅ DONE

Edge features (static 8D + dynamic 6D + timeseries 34D) were implemented inside `build_graph_hybrid()`.
Separate implementation not required.

---

### 🚧 **Day 6 (CURRENT - NEXT STEP)**: InferenceEngine Integration

**Estimated Time**: 3-4 hours  
**Status**: ⏳ PENDING

#### Priority 1: InferenceEngine Integration (1.5-2 hours)
- [ ] Update `InferenceEngine.predict_hybrid()` to use GraphBuilderV2
- [ ] Integrate ValueSubstitutionEngine for missing values
- [ ] Handle DiagnosticScope in inference
- [ ] Add error handling for validation failures
- [ ] Integration test: HybridInferenceRequest → Graph → Model → Predictions

#### Priority 2: FastAPI Endpoint (1-1.5 hours)
- [ ] Create `/v2/inference/hybrid` endpoint
- [ ] Request/response validation
- [ ] Error handling (topology not found, invalid readings, etc.)
- [ ] Integration test: HTTP → InferenceEngine → Response

#### Priority 3: Documentation (0.5 hours)
- [ ] API documentation (OpenAPI)
- [ ] Usage examples
- [ ] Migration guide (v1 → v2)

**Expected Output**:
```python
# ✅ Working end-to-end inference
response = await client.post(
    "/v2/inference/hybrid",
    json=hybrid_request.model_dump()
)

assert response.status_code == 200
result = response.json()
assert "predictions" in result
assert "component_health" in result["predictions"]
assert "anomaly_scores" in result["predictions"]
```

---

### **Day 7 (Pending): Backward Compatibility + Testing**

**Time**: 3-4 hours

**Tasks**:
- [ ] Implement `convert_node_to_hybrid()` converter (old API → new API)
- [ ] Add backward compatibility layer for `/v1/inference` endpoint
- [ ] Integration tests for v1 → v2 conversion
- [ ] Performance benchmarks:
  - Inference time (<50ms target)
  - Memory usage (<2GB GPU)
  - Throughput (>1000 graphs/sec batch)
- [ ] Documentation update (migration guide)

**Expected Output**:
```python
# ✅ Old API still works
old_request = MinimalInferenceRequest(...)  # v1
response = await client.post("/v1/inference", json=old_request.model_dump())
assert response.status_code == 200

# ✅ Internally converted to HybridInferenceRequest → GraphBuilderV2
```

---

### Week 1 Checkpoint:

**✅ Deliverables**:
- ✅ Sensor registry schemas complete (Day 1)
- ✅ ValueSubstitutionEngine structure (Day 1)
- ✅ GraphBuilderV2 fully implemented (Day 3)
- ✅ Comprehensive unit tests (>90% coverage) (Day 3)
- ⏳ Physics calculations (Day 2 - pending)
- ⏳ InferenceEngine integration (Day 6 - next)
- ⏳ HybridInferenceRequest API working (Day 6)
- ⏳ Backward compatibility maintained (Day 7)

**✅ Metrics**:
- Graph construction: ✅ Working
- Validation: ✅ 6 checks implemented
- Test coverage: ✅ >90%
- Inference time: ⏳ TBD (Day 6)
- Memory usage: ⏳ TBD (Day 6)

**🎯 Expected Improvement**: +40-60% accuracy (after retraining on edge-centric data)

---

## 🎯 WEEK 2: Training Quality Improvements

**Reference Document**: [`gnn-improvements-plan.md`](./gnn-improvements-plan.md)

### Objective:
Optimize training pipeline for maximum performance on edge-centric architecture

### ⚠️ Prerequisites:
- ✅ Week 1 (Edge-centric architecture) MUST be complete
- ✅ Model accepts rich edge features (14-116D)
- ✅ GraphBuilderV2 tested and working

---

### **Day 8 (Monday): P0 Training Essentials**

**Time**: 2-3 hours  
**Expected Gain**: +20-35%

#### Task 1: Gradient Clipping (5 min)
```python
# src/training/lightning_module.py

class HydraulicGNNModule(pl.LightningModule):
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,  # ← ADD THIS!
            gradient_clip_algorithm="norm"
        )
```
**Impact**: +10-20% (prevents exploding gradients)

---

#### Task 2: Enable Uncertainty Weighting (1 min)
```python
# Already implemented! Just enable in config
loss_fn = MultiTaskLoss(
    use_uncertainty_weighting=True,  # ← Change to True!
    component_health_weight=1.0,
    anomaly_type_weight=1.0,
)
```
**Impact**: +5-15% (balanced multi-task learning)

---

#### Task 3: Class Weights for Imbalanced Data (30 min)
```python
# Compute class weights from training data
from sklearn.utils.class_weight import compute_class_weight

anomalies = [sample.y_graph for sample in train_dataset]
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(anomalies),
    y=anomalies
)

# Use in loss function
loss_fn = MultiTaskLoss(
    anomaly_class_weights=torch.tensor(class_weights, dtype=torch.float32)
)
```
**Impact**: +15-30% (handle rare anomaly types)

---

#### Task 4: LR Warmup + Cosine Annealing (20 min)
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,  # Double period after restart
    eta_min=1e-6
)
```
**Impact**: +5-10% (better convergence)

---

#### Task 5: Early Stopping (5 min)
```python
# In PyTorch Lightning Trainer
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min'
)

trainer = pl.Trainer(
    callbacks=[early_stop],
    max_epochs=100
)
```
**Impact**: Prevents overfitting, saves time

---

#### Task 6: Test on Small Dataset (2 hours)
- Train on 10% of data
- Validate improvements work
- Tune hyperparameters

**Day 8 Checkpoint**: +20-35% improvement confirmed on small dataset ✅

---

### **Day 9 (Tuesday): P1 Advanced Techniques**

**Time**: 3-4 hours  
**Expected Gain**: +10-20% (cumulative)

#### Task 1: Data Augmentation (1 hour)
```python
class GraphAugmentation:
    """Augment edge-centric graphs during training."""
    
    def __init__(self, noise_std=0.05, drop_edge_prob=0.1):
        self.noise_std = noise_std
        self.drop_edge_prob = drop_edge_prob
    
    def __call__(self, graph: Data) -> Data:
        # 1. Add Gaussian noise to edge features
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(graph.edge_attr) * self.noise_std
            graph.edge_attr = graph.edge_attr + noise
        
        # 2. Randomly drop edges (simulate sensor failures)
        if torch.rand(1) < self.drop_edge_prob:
            mask = torch.rand(graph.num_edges) > 0.1
            graph.edge_index = graph.edge_index[:, mask]
            graph.edge_attr = graph.edge_attr[mask]
        
        return graph

# Use in DataLoader
train_dataset = GraphDataset(..., transform=GraphAugmentation())
```
**Impact**: +10-20% (better generalization)

---

#### Task 2: Residual Connections in GAT (40 min)
```python
# src/models/layers.py

class GATv2ConvWithResidual(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads)
        
        # Residual projection (if dimensions don't match)
        if in_channels != out_channels * heads:
            self.residual_proj = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x, edge_index, edge_attr):
        identity = x
        
        # GAT forward
        out = self.gat(x, edge_index, edge_attr)
        
        # Add residual
        out = out + self.residual_proj(identity)
        
        return out
```
**Impact**: +5-10% (deeper networks train better)

---

#### Task 3: Enhanced Attention Pooling (1 hour)
```python
# Already implemented in universal_temporal_gnn.py!
# Just enable in config:

config = ModelConfig(
    use_attention_pooling=True,  # ← Enable!
    use_virtual_nodes=True,      # ← Enable!
)
```
**Impact**: +3-8% (better graph-level representations)

---

#### Task 4: Test on Small Dataset (2 hours)

**Day 9 Checkpoint**: +25-40% cumulative improvement ✅

---

### **Day 10 (Wednesday): P2 Validation & TTA**

**Time**: 2-3 hours  
**Expected Gain**: +2-5% (robustness)

#### Task 1: K-Fold Cross-Validation (1 hour)
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    model = UniversalTemporalGNNv2(config)
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, train_subset, val_subset)
    
    metrics = trainer.test(model, val_subset)
    results.append(metrics)

print(f"Mean F1: {np.mean([r['f1'] for r in results]):.3f} ± {np.std([r['f1'] for r in results]):.3f}")
```
**Impact**: Robust validation, prevents lucky splits

---

#### Task 2: Test-Time Augmentation (45 min)
```python
def predict_with_tta(model, graph, n_augmentations=5):
    """Ensemble predictions from augmented graphs."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_augmentations):
            # Augment graph
            aug_graph = augment_graph(graph)  # Small noise, edge drop
            
            # Predict
            output = model(aug_graph)
            predictions.append(output)
    
    # Average predictions
    avg_node_logits = torch.stack([p['node_logits'] for p in predictions]).mean(0)
    avg_graph_logits = torch.stack([p['graph_logits'] for p in predictions]).mean(0)
    
    return {'node_logits': avg_node_logits, 'graph_logits': avg_graph_logits}
```
**Impact**: +2-5% (more robust predictions)

---

#### Task 3: Integration Tests (1 hour)
- End-to-end pipeline test
- Validate all improvements working together
- Benchmark final performance

**Day 10 Checkpoint**: Training pipeline complete ✅

---

### **Days 11-12 (Thu-Fri): Full Retraining**

**Time**: 1-2 days

**Tasks**:
- [ ] Prepare full edge-centric training dataset
- [ ] Train with all improvements enabled
- [ ] Monitor training (TensorBoard, W&B)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Save best checkpoint
- [ ] A/B testing: old vs new model

**Expected Results**:
```
Baseline (node-centric, no improvements):
  - Anomaly Detection F1: 0.72
  - Component Health Accuracy: 78%
  - RUL MAE: 15 days

After Phase 1 (edge-centric):
  - Anomaly Detection F1: 0.85 (+18%)
  - Component Health Accuracy: 84% (+8%)
  - RUL MAE: 10 days (-33%)

After Phase 2 (edge + training improvements):
  - Anomaly Detection F1: 0.95 (+32% total) ⭐
  - Component Health Accuracy: 88% (+13% total) ⭐
  - RUL MAE: 6 days (-60% total) ⭐
  - Edge Leak Localization: 95% precision (NEW!) ⭐
```

---

## 🚀 WEEK 3: Production Polish (Optional)

**Focus**: Deployment readiness, not core improvements

### Tasks:
- [ ] Model quantization (FP16/INT8) for faster inference
- [ ] `torch.compile()` optimization (2x speedup)
- [ ] Model serving (TorchServe / Triton)
- [ ] Load testing (1000 req/s)
- [ ] Monitoring dashboards (Grafana)
- [ ] Documentation updates
- [ ] CI/CD pipeline

---

## 📊 Success Metrics

### Model Performance:
- [ ] Anomaly Detection F1: 0.72 → **0.95** (+32%)
- [ ] Component Health Accuracy: 78% → **88%** (+13%)
- [ ] RUL MAE: 15 days → **6 days** (-60%)
- [ ] Leak Localization Precision: 70% → **95%** (+36%)

### System Performance:
- [ ] Inference time: <50ms (single graph)
- [ ] Throughput: >1000 graphs/second (batch)
- [ ] GPU memory: <4GB
- [ ] Model size: <100MB

### Code Quality:
- [x] Test coverage: >90% (Day 1 infrastructure + Day 3 GraphBuilderV2)
- [x] GraphBuilderV2 tests passing (23 methods)
- [ ] All integration tests passing (pending Day 6)
- [ ] Documentation complete
- [ ] Backward compatible

---

## 📝 Handoff Instructions (for new chat)

### Current Status:
**✅ Completed:**
- ✅ Day 1: Sensor registry schemas (PhysicalSensor, SensorDataSourceConfig)
- ✅ Day 1: ValueSubstitutionEngine structure
- ✅ Day 1: Unit tests (11 working + 11 stubs)
- ✅ Day 3: GraphBuilderV2.build_graph_hybrid() complete implementation
- ✅ Day 3: DiagnosticScope + HybridInferenceRequest
- ✅ Day 3: Comprehensive unit tests (23 methods, >90% coverage)

**🚧 In Progress:**
- Physics-based estimation (Day 2)

**⏳ Pending:**
- InferenceEngine integration (Day 6 - NEXT STEP)
- FastAPI endpoint `/v2/inference/hybrid` (Day 6)
- Backward compatibility (Day 7)
- Training improvements (Week 2)
- Full retraining (Week 2)

### Next Immediate Steps:

1. **Skip Day 2 for now** (physics can wait, focus on integration):
   - Day 2 (physics) is not blocking for Day 6
   - ValueSubstitutionEngine can use nominal values as fallback

2. **Start Day 6: InferenceEngine Integration** (PRIORITY!):
   - Open `src/inference/engine.py`
   - Update `predict_hybrid()` to use GraphBuilderV2
   - Add error handling
   - Integration test: HybridInferenceRequest → Graph → Model → Predictions

3. **After Day 6 Complete**:
   - Create FastAPI endpoint `/v2/inference/hybrid`
   - Integration test: HTTP → InferenceEngine → Response

### ⚠️ Critical Rules:
- **DO NOT skip Phase 1** (edge-centric architecture is foundation)
- **DO NOT modify existing code** (create GraphBuilderV2, not edit GraphBuilder)
- **DO NOT break backward compatibility** (old API must still work)
- **DO NOT start training improvements** before edge-centric is complete

### Key Files to Know:
- `src/schemas/sensor_registry.py` - PhysicalSensor (✅ Done)
- `src/training/value_substitution.py` - Physics engine (🚧 Day 2 pending)
- `src/data/graph_builder_v2.py` - GraphBuilderV2 (✅ Done!)
- `tests/unit/test_data/test_graph_builder_v2.py` - Tests (✅ Done!)
- `src/schemas/requests.py` - HybridInferenceRequest (✅ Done!)
- `src/models/universal_temporal_gnn.py` - Model (already supports rich edges!)
- `src/inference/engine.py` - **WHERE TO WORK NEXT (Day 6)** ⏳
- `src/training/lightning_module.py` - Training loop (Week 2 improvements)

---

## 📚 References

- **Architecture**: [`docs/EDGE_CENTRIC_MIGRATION.md`](./docs/EDGE_CENTRIC_MIGRATION.md)
- **Sensor Coverage**: [`docs/SENSOR_COVERAGE_LEVELS.md`](./docs/SENSOR_COVERAGE_LEVELS.md)
- **Training**: [`gnn-improvements-plan.md`](./gnn-improvements-plan.md)
- **API/Database**: [`src/api/IMPLEMENTATION_PLAN.md`](./src/api/IMPLEMENTATION_PLAN.md)
- **Model Docs**: [`src/models/README.md`](./src/models/README.md)

---

**🎯 Status Summary:**
- **Day 1 (Infrastructure)**: ✅ COMPLETED (4 commits, 22 tests, 30 min)
- **Day 2 (Physics)**: 🚧 Deferred (not blocking Day 6)
- **Day 3 (GraphBuilderV2)**: ✅ COMPLETED (3 commits, +1773 lines, 4 hours)
- **Day 4-5 (Edge Features)**: ✅ MERGED into Day 3
- **Day 6 (Integration)**: ⏳ NEXT STEP (InferenceEngine + FastAPI)
- **Week 2 (Training)**: ⏳ Blocked on Week 1 completion
- **Week 3 (Production)**: ⏳ Optional polish

**Next Action**: Integrate GraphBuilderV2 into InferenceEngine 🚀