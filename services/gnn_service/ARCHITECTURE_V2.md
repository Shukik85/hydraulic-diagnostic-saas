# 🏗️ GNN Hydraulic Diagnostics V2: Production Architecture

**Created:** January 3, 2026  
**Version:** 1.0.0  
**Status:** SPECIFICATION → IMPLEMENTATION

---

## 🎯 CORE PRINCIPLES

1. **Zero tolerance for TODO/fallback code** — Every line production-ready
2. **Single responsibility** — Each module does ONE thing correctly
3. **No orphan code** — Every file actively used or removed
4. **Transparent contracts** — Type hints + docstrings everywhere
5. **Fail-fast semantics** — Exception raising, NEVER silent fallbacks
6. **Observable by default** — Structured logging + metrics from start

---

## 📐 DIRECTORY STRUCTURE

```
services/gnn_service/src_v2/          # ← Brand new, clean slate
├── __init__.py
│
├── core/                              # Immutable, production primitives
│   ├── __init__.py
│   ├── config.py                      # TrainingConfig, DataConfig (Pydantic)
│   ├── exceptions.py                  # Custom exception hierarchy
│   ├── logging.py                     # Structured logging setup
│   ├── constants.py                   # Magic numbers, enums
│   └── types.py                       # TypeAliases for tensors, arrays
│
├── data/                              # Data pipeline (Raw UCI → Graphs)
│   ├── __init__.py
│   │
│   ├── sources/                       # UCI data source adapter
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract DataSource
│   │   ├── uci_loader.py              # Load raw UCI .txt files (100 Hz)
│   │   └── validator.py               # Data quality checks
│   │
│   ├── processing/                    # Feature engineering pipeline
│   │   ├── __init__.py
│   │   ├── resampler.py               # 100 Hz → 10 Hz downsampling
│   │   ├── normalizer.py              # Z-score normalization (train-only)
│   │   ├── feature_extractor.py       # 17 channels → 14D edge features
│   │   ├── semisynthetic.py           # Derive features from sensor physics
│   │   └── quality_gates.py           # Data validation, outlier detection
│   │
│   ├── graphs/                        # Graph construction
│   │   ├── __init__.py
│   │   ├── builder.py                 # Unified graph builder (wraps GraphBuilderV2)
│   │   ├── validator.py               # Graph structure validation
│   │   └── collator.py                # PyG collate function
│   │
│   └── loaders.py                     # Dataset + DataLoader factories
│
├── models/                            # GNN models
│   ├── __init__.py
│   ├── gnn.py                         # Main GNN (GAT + LSTM + MLP)
│   └── components/                    # Reusable blocks
│       ├── __init__.py
│       ├── gat_layer.py
│       ├── lstm_layer.py
│       └── mlp.py
│
├── training/                          # Training loop (raw PyTorch)
│   ├── __init__.py
│   ├── losses.py                      # Loss functions
│   ├── metrics.py                     # Evaluation metrics (F1, ROC-AUC)
│   ├── optimizer.py                   # LR scheduling, warmup
│   ├── trainer.py                     # Main training loop
│   └── checkpointing.py               # Save/load with version metadata
│
├── inference/                         # Inference pipeline
│   ├── __init__.py
│   ├── engine.py                      # InferenceEngine (wraps GraphBuilderV2)
│   ├── predictor.py                   # High-level prediction API
│   └── monitoring.py                  # Latency/accuracy tracking
│
├── observability/                     # Logging + metrics + tracing
│   ├── __init__.py
│   ├── logging.py                     # Structured JSON logging
│   ├── metrics.py                     # Prometheus metrics
│   ├── tracing.py                     # OpenTelemetry instrumentation
│   └── health.py                      # Liveness/readiness checks
│
└── cli/                               # Command-line tools
    ├── __init__.py
    ├── train.py                       # python -m src_v2.cli.train
    ├── evaluate.py                    # python -m src_v2.cli.evaluate
    ├── export.py                      # Export to ONNX/TorchScript
    ├── inspect.py                     # Inspect data, model, checkpoints
    └── preprocess.py                  # Preprocess raw UCI → 10Hz parquet
```

---

## 🔄 DATA PIPELINE: UCI → Training

### Input: Raw UCI Hydraulic Dataset (530 MB)

```
services/gnn_service/data/raw_real_dataset/
├── PS1.txt, PS2.txt, ..., PS6.txt    # Pressure (6 channels)
├── TS1.txt, ..., TS4.txt             # Temperature (4 channels)
├── FS1.txt, FS2.txt                  # Flow rate (2 channels)
├── VS1.txt                            # Vibration (1 channel)
├── SE.txt                             # Solenoid state (1 channel)
├── CE.txt, CP.txt                     # Cumulative energy/power (2 channels)
└── documentation.txt                  # Labels: cooler, valve, pump, accumulator
```

**Characteristics:**
- 100 Hz sampling rate
- ~2600 cycles (60-second windows)
- 17 sensors total
- 4×3 multi-label targets (cooler, valve, pump, accumulator × 3 severity)
- <0.1% missing data, minimal outliers

---

### Pipeline Stages

#### Stage 1: UCI Loader (sources/uci_loader.py)

```python
class UCIDataSource:
    """Load raw UCI .txt files, validate, yield EquipmentCycle objects"""
    
    def __init__(self, data_dir: Path):
        self.channels = self._load_channels()  # 17 arrays
        self.labels = self._load_labels()      # 4×3 matrix
    
    def load_cycles(self, start_idx: int, end_idx: int) -> Iterator[EquipmentCycle]:
        """
        Yields cycles with proper timestamps and labels
        
        Each cycle:
        - samples @ 100 Hz: shape (7000, 17) for 60 seconds
        - timestamp: datetime of cycle start
        - labels: Dict[str, int] — cooler_state, valve_state, etc.
        - metadata: equipment_id, cycle_number
        """
    
    def validate(self) -> ValidationReport:
        """Check for data quality issues"""
```

**Key Decision:** Use **fixed 60-second windows** (simpler, deterministic) vs. **solenoid-triggered** (more realistic).  
→ Start with fixed windows, ~2600 cycles.

---

#### Stage 2: Resampling (processing/resampler.py)

```
100 Hz × 7000 samples/cycle = 700 KB/cycle
                    ↓ (downsample by 10×)
10 Hz × 700 samples/cycle = 70 KB/cycle

10× compression → GPU-friendly sequence length
```

```python
class Resampler:
    """100 Hz → 10 Hz via simple averaging (no IIR artifacts)"""
    
    def resample(self, cycle_raw: np.ndarray) -> np.ndarray:
        """
        Input: [7000, 17]
        Output: [700, 17]
        
        Method: Simple averaging (scipy.signal.resample alternative)
        """
```

---

#### Stage 3: Quality Gates (quality_gates.py)

```python
class QualityValidator:
    """Detect corrupted cycles BEFORE training"""
    
    def check_missing_data(self, cycle: np.ndarray) -> bool:
        """Reject if >1% NaN"""
    
    def check_outliers(self, cycle: np.ndarray) -> bool:
        """Remove if pressure > 500 bar, temp > 80°C, etc."""
    
    def check_monotonicity(self, cycle: np.ndarray) -> bool:
        """Cumulative energy must be monotonic increasing"""
    
    def get_quality_score(self, cycle: np.ndarray) -> float:
        """0-1 score, filter out <0.8"""
```

**Gate Rules (inherited from RAW_DATA_QUALITY_ASSESSMENT.md):**
- Missing data: <0.1% allowed
- Outliers: >3σ rejected
- Physical limits: pressure 0-350 bar, temp 0-80°C
- Monotonicity: cumulative columns must increase

---

#### Stage 4: Normalization (processing/normalizer.py)

```python
class Normalizer:
    """Z-score normalization (train-only statistics)"""
    
    def fit(self, train_cycles: np.ndarray) -> None:
        """Compute μ, σ from TRAINING data only"""
        self.mean = np.mean(train_cycles, axis=(0, 1))
        self.std = np.std(train_cycles, axis=(0, 1))
        # ⚠️ CRITICAL: Validation/test use train stats!
    
    def transform(self, cycle: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        return (cycle - self.mean) / (self.std + 1e-8)
```

**NO data leakage:** Fit on train set, apply to val/test.

---

#### Stage 5: Semisynthetic Feature Extraction (processing/semisynthetic.py)

**Philosophy:** 17 raw channels → **14D edge features** (static + dynamic derived)

```python
class SemisyntheticFeatureExtractor:
    """Transform raw sensor readings → edge-centric features via physics"""
    
    def extract_features(self, 
        cycle: EquipmentCycle,  # 17 channels, 700 samples @ 10 Hz
        edges: Dict[str, EdgeDefinition]  # topology
    ) -> Dict[str, np.ndarray]:  # {edge_name: [700, 14]}
        """
        For each hydraulic line (edge), compute:
        
        STATIC (8D, constant per cycle):
        1. nominal_pressure (bar) — line's setpoint
        2. nominal_flow (L/min) — expected rate
        3. nominal_temp (°C) — normal operating temp
        4. line_length (m) — physical parameter
        5. line_diameter (mm)
        6. fluid_viscosity (cSt) — tabular lookup
        7. design_max_pressure (bar)
        8. design_max_temp (°C)
        
        DYNAMIC (6D, time-varying):
        9. pressure_mean (bar) — avg over cycle
        10. pressure_std (bar) — variability
        11. flow_rate_mean (L/min)
        12. flow_rate_std (L/min)
        13. temp_trend (°C over 60s) — slope
        14. efficiency_ratio = flow / (pressure_drop + ε)
        
        All normalized to [0, 1] range
        """
```

**Semisynthetic Rationale:**
- **Static:** From hydraulic system topology (FIXED once, tabular)
- **Dynamic:** Computed from sensors (varies per cycle)
- Reduces information overload (not 700×17 raw values)
- Aligns with hydraulic physics understanding

---

#### Stage 6: Graph Construction (graphs/builder.py)

```python
class GraphBuilder:
    """Unified graph constructor using GraphBuilderV2 logic"""
    
    def build(self, 
        cycle: EquipmentCycle,
        edge_features: Dict[str, np.ndarray],  # from extractor
        scope: DiagnosticScope = "full"
    ) -> Data:
        """
        Returns PyG Data:
        - x: [N, 29] node features (components: pumps, motors, valves, etc.)
        - edge_index: [2, E] connectivity matrix
        - edge_attr: [E, 14] features per edge
        - y: [4, 3] multi-label targets (cooler, valve, pump, accumulator)
        - metadata: cycle_id, timestamp
        
        Internally: Wraps GraphBuilderV2.build_graph_hybrid()
        """
```

---

#### Stage 7: DataLoader Factory (loaders.py)

```python
def create_dataloader(
    dataset: Dataset,
    config: DataConfig,
    split: Literal["train", "val", "test"] = "train"
) -> DataLoader:
    """
    Returns PyTorch DataLoader with:
    - collate_fn=batch_graphs (PyG-compatible)
    - num_workers={2 if train else 0} (avoid I/O contention)
    - shuffle={True if train else False}
    - drop_last={True if train else False}
    """

def create_train_val_test_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Temporal split: no data leakage"""
```

---

## 📦 KEY MODULES

### (1) EquipmentCycle — Immutable Data Class

```python
@dataclass(frozen=True)
class EquipmentCycle:
    """Single 60-second equipment measurement cycle"""
    cycle_id: str
    timestamp: datetime
    
    # Raw sensors [700, 17] @ 10 Hz
    sensor_data: np.ndarray  # After resampling + normalization
    
    # Labels [4, 3] multi-label
    cooler_state: int  # 0=ok, 1=reduced, 2=failed
    valve_state: int
    pump_leakage: int
    accumulator_state: int
    
    # Metadata for tracing
    raw_sample_count: int  # original @ 100 Hz
    quality_score: float  # 0-1, gating threshold
```

### (2) TrainingConfig — Hyperparameters

```python
@dataclass
class TrainingConfig:
    """Central training configuration"""
    num_epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    
    # Model
    gat_heads: int = 8
    lstm_hidden: int = 64
    mlp_hidden: int = 128
    dropout: float = 0.2
    
    # Checkpoint
    save_every_n_epochs: int = 10
    checkpoint_dir: Path = Path("checkpoints")
```

### (3) ValidationReport — Quality Gates Output

```python
@dataclass
class ValidationReport:
    """Result of data quality checks"""
    total_cycles: int
    valid_cycles: int
    rejected_cycles: List[Tuple[str, str]]  # (cycle_id, reason)
    
    missing_data_issues: int
    outlier_issues: int
    monotonicity_issues: int
    
    passed_rate: float  # valid / total
    
    def to_dict(self) -> dict:
        """For logging/monitoring"""
```

---

## ⚠️ EXCEPTION HIERARCHY (core/exceptions.py)

```python
class DataPipelineError(Exception):
    """Base for all data pipeline errors"""
    pass

class UCILoadingError(DataPipelineError):
    """Failed to load UCI files"""
    pass

class ValidationError(DataPipelineError):
    """Data quality check failed"""
    pass

class NormalizationError(DataPipelineError):
    """Normalization stats not fitted"""
    pass

class FeatureExtractionError(DataPipelineError):
    """Edge feature computation failed"""
    pass

class GraphConstructionError(DataPipelineError):
    """Graph building failed (invalid topology, missing features)"""
    pass

class TrainingError(Exception):
    """Base for training errors"""
    pass

class CheckpointError(TrainingError):
    """Checkpoint save/load failed"""
    pass
```

**NO silent failures:** Every error raised explicitly with context.

---

## 🧪 QUALITY GATES (Tests Must Pass)

### Unit Tests (>90% coverage)

```python
# tests/data/test_uci_loader.py
def test_uci_loader_loads_all_channels():
    loader = UCIDataSource(data_dir)
    assert loader.num_cycles == 2600
    assert loader.num_channels == 17

def test_uci_loader_rejects_corrupted():
    # Manually corrupt PS1.txt
    with pytest.raises(UCILoadingError):
        loader.load_cycles(...)

# tests/data/test_resampler.py
def test_resampler_100hz_to_10hz():
    raw = np.random.randn(7000, 17)
    resampled = Resampler().resample(raw)
    assert resampled.shape == (700, 17)

# tests/data/test_feature_extractor.py
def test_semisynthetic_features_14d():
    cycle = create_test_cycle()
    extractor = SemisyntheticFeatureExtractor()
    features = extractor.extract_features(cycle, edges)
    for edge_name, edge_feat in features.items():
        assert edge_feat.shape == (700, 14)
        assert np.all(np.isfinite(edge_feat))  # no NaN/inf

# tests/data/test_graph_builder.py
def test_graph_builder_produces_valid_data():
    cycle = create_test_cycle()
    features = extractor.extract_features(cycle, edges)
    graph = builder.build(cycle, features)
    
    assert graph.x.shape[1] == 29  # node features
    assert graph.edge_attr.shape[1] == 14
    assert torch.all(torch.isfinite(graph.edge_attr))
    assert graph.y.shape == (4, 3)  # multi-label
```

### Integration Test (E2E)

```python
# tests/data/test_pipeline_e2e.py
def test_full_pipeline_uci_to_dataloader():
    """UCI files → normalized cycles → graphs → DataLoader"""
    
    loader = UCIDataSource(data_dir)
    dataset = HydraulicGraphDataset(loader, feature_extractor, graph_builder)
    
    train_loader, val_loader, test_loader = create_train_val_test_split(dataset)
    
    # Check first batch
    batch = next(iter(train_loader))
    assert batch.x.shape[0] > 0  # nodes
    assert batch.num_graphs == 32  # batch size
    assert batch.edge_attr.shape[1] == 14
    
    # Check no leakage
    train_ids = set(dataset.cycle_ids[:int(0.7*len(dataset))])
    val_ids = set(dataset.cycle_ids[int(0.7*len(dataset)):])
    assert len(train_ids & val_ids) == 0
```

---

## 📊 SUCCESS CRITERIA

### Code Quality
- [ ] 100% type hints (`mypy --strict`)
- [ ] >90% test coverage (`pytest --cov`)
- [ ] **ZERO** TODO/FIXME/XXX in production code
- [ ] **ZERO** `if ... logger.warning; fallback_code()`
- [ ] All public functions have docstrings + examples

### Data Quality
- [ ] UCIDataSource validates all 2600 cycles
- [ ] <1% cycles rejected by quality gates
- [ ] Normalization stats computed from train set only
- [ ] Temporal split enforced (no leakage)

### Performance
- [ ] UCI → 10Hz cycles: <5 minutes (1 engineer, 1 CPU)
- [ ] 2600 cycles → DataLoader: <2 seconds
- [ ] Train forward pass (batch 32): <100ms (GPU)

### Reproducibility
- [ ] Seed fixed for random_split
- [ ] Checkpoint includes: model weights, optimizer state, epoch, best_val_loss, edge_in_dim
- [ ] Exact same training results with same seed

---

## 🔗 INTEGRATION WITH EXISTING CODE

### USE (Reuse, don't rewrite)
- ✅ **GraphBuilderV2** (src/data/graph_builder_v2.py)  
  → Wrap in graphs/builder.py

- ✅ **Schemas** (src/schemas/*.py)  
  → Import HybridInferenceRequest, DiagnosticScope

- ✅ **UCI Raw Data** (data/raw_real_dataset/)  
  → Load via sources/uci_loader.py

### DELETE (Don't bring forward)
- ❌ src/data/dataset.py (HydraulicGraphDataset with TODO)
- ❌ src/data/graph_builder.py (old homogeneous logic)
- ❌ src/training/lightning_module.py (not using Lightning)
- ❌ src/data/adapters/ (orphaned code)
- ❌ src/mapping/apply_mapping.py (merged into feature_extractor)

### REFACTOR (Later phases)
- 🔄 src/data/loader.py → adapt signatures to src_v2
- 🔄 src/training/train_temporal.py → reference implementation

---

## 📋 PHASE 1: DATA PIPELINE (WEEK 1)

### Deliverables
- [ ] **core/** — All exceptions, config, types
- [ ] **data/sources/uci_loader.py** — Load 2600 cycles from raw UCI
- [ ] **data/processing/** — Resample, normalize, extract, validate
- [ ] **data/graphs/builder.py** — Unified graph construction
- [ ] **data/loaders.py** — DataLoader + split factories
- [ ] **tests/** — >90% coverage, E2E integration test

### Success Condition
```bash
pytest tests/data/ -v --cov=src_v2/data
# Output: 
# PASSED: 45 tests, 92% coverage
# No errors, no TODO comments
```

### Acceptance Test
```python
def test_phase1_complete():
    loader = UCIDataSource("data/raw_real_dataset")
    cycles = list(loader.load_cycles(0, 2600))
    assert len(cycles) == 2600
    
    dataset = HydraulicGraphDataset(cycles)
    train_loader, val_loader, _ = create_train_val_test_split(dataset)
    
    batch = next(iter(train_loader))
    assert batch.x.shape == (nodes, 29)
    assert batch.edge_attr.shape == (edges, 14)
    assert batch.y.shape == (batch_size * 4, 3)  # multi-label
    
    print("✅ Phase 1 COMPLETE: UCI → Training-Ready DataLoader")
```

---

## 🚀 FUTURE PHASES (Not in Scope)

**Phase 2 (Week 2):** GNN model + training loop  
**Phase 3 (Week 3):** Inference + API  
**Phase 4 (Week 4):** Observability + monitoring

---

## 📝 DECISIONS LOCKED

✅ **Data Source:** UCI Hydraulic (raw_real_dataset/)  
✅ **Data Format:** 10 Hz resampled, normalized, 14D features  
✅ **Graph Architecture:** Edge-centric (uses GraphBuilderV2)  
✅ **Training Framework:** Raw PyTorch (no Lightning overhead)  
✅ **Checkpoint Format:** Versioned with metadata  
✅ **Error Handling:** Fail-fast, explicit exceptions  
✅ **Code Quality:** 100% type hints, >90% tests, zero TODO  

---

**Next:** Approve this architecture → Start Phase 1 implementation
