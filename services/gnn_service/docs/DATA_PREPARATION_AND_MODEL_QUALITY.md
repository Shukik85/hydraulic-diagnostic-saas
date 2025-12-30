# 📒 Data Preparation & Model Quality Expectations

**Phase 3.2+ Training Strategy Based on Real Hydraulic Cycles**

---

## 📊 **Data Characteristics Analysis**

### **What We See in cycle_00000_10hz_stacked.jpg**

This is a **real hydraulic test cycle** (~65 seconds) with 16 sensor channels at 10Hz sampling:

#### **Channel Breakdown (Physical Interpretation)**

| Row # | Likely Signal | Range | Physics |
|-------|---------------|-------|----------|
| **1** | Pump outlet pressure (P1) | 150-190 bar | Main supply line; sharp spikes = load engagement |
| **2** | Tank/return pressure (P2) | 0-100 bar | Return manifold pressure; mostly low |
| **3** | Line ΔP (P3-P4) | 0-7.5 bar | Pressure drop across valve/actuator |
| **4** | Pressure feedback (P4) | ~0 bar | Tank/reference (flat) |
| **5** | Electrical signal (PSI) | ±0.05 (flat) | Proportional spool current input? |
| **6** | Coolant return temp (T1) | 9.7-9.9 bar | **Temperature, not pressure!** Warm side |
| **7** | Tank temp (T2) | 9.6-9.8 bar | **Temperature** Cool side; steady |
| **8** | Motor speed (EMS) | 2400-2800 RPM | Engine RPM; stable cruise |
| **9** | Actuator position (FJ) | 0-10 L/min | **Flow rate?** Variable, matches load profile |
| **10** | Pump outlet flow (L2) | 0-10 L/min | Secondary supply; spiky = pressure relief |
| **11** | Viscosity/density (FS) | 10.2-10.4 | Fluid property index (non-dimensional) |
| **12** | Load pressure (TS) | 35-36 °C | **Temperature!** Load-side thermal signature |
| **13** | Accumulator pressure (TS2) | 40.8-41.2 bar | Accumulator stored energy |
| **14** | Tank sump level (TS3) | 38.4-38.6 °C | **Temperature.** Heat dissipation monitor |
| **15** | Fluid condition (FS2) | 31-33 °C | **Temperature.** System thermal state |
| **16** | Pump intake valve (VS) | 0-60 % | **Valve position.** Suction control/unload |

---

## 🔍 **Key Observations from the Data**

### **1. Strong Operational Patterns**

```
0-10s:  Startup transient
  ✓ P1 spikes: pump pressure ramps to 150-190 bar
  ✓ FJ (actuator): ramps from 0 → max (load engagement)
  ✓ TS (load temp): stable baseline
  ✓ VS (pump unload): drops sharply (goes on load)

10-40s: Steady operation
  ✓ P1 oscillates ±10 bar (pressure relief chatter)
  ✓ FJ constant (~5-8 L/min load flow)
  ✓ T1, T2 drift slowly upward (system heating)
  ✓ TS2 (accumulator) stable (~41 bar)
  ✓ VS creeps up (unload valve drift?)

40-65s: Cool-down / unload
  ✓ P1 drops to 150 bar baseline
  ✓ FJ decays to zero (actuator retracting)
  ✓ T1, T2 plateau (peak temperature reached)
  ✓ VS rises to 60% (full unload)
```

### **2. Noise vs Signal Characteristics**

| Signal Type | Examples | Noise Level | Training Impact |
|------------|----------|-------------|------------------|
| **Slow trends** | T1, T2, TS, TS3, TS2 | Very low (~0.05°C/bar) | Easy to learn; good for RUL |
| **Steady state** | FS, FS2 (fluid property) | Very low (~0.2 units) | Stable baseline reference |
| **High-frequency** | P1, FJ, L2 | Medium (~10% of range) | Valve chatter, pump ripple |
| **On/off** | VS (0-60%) | Low (step changes) | Clear operational phases |
| **Electrical** | PSI (±0.05) | Very high (noisy) | **May need filtering or exclusion** |

### **3. Cycle Duration & Sampling**

- **Duration:** ~65 seconds per cycle
- **Sampling rate:** 10 Hz
- **Samples per cycle:** 650 samples
- **Typical dataset:** 500-5000 cycles → **325K - 3.25M samples**

**Implication:** 
- ✅ Sufficient temporal context (6-second windows = 60 timesteps)
- ✅ Multiple operational phases per cycle (startup, steady, cool-down)
- ⚠️ Class imbalance: most of cycle is "healthy" operation

---

## 📦 **Training Data Preparation Strategy**

### **Phase 1: Raw Data to Features (GraphBuilderV2)**

**Goal:** Convert 16-channel raw samples → node/edge features

#### **1.1 Sensor Selection & Grouping**

```python
# EDGE SENSORS (Line-level: signals flowing through hydraulic network)
edge_sensors = {
    'pressure_inlet': ['P1', 'P2'],          # Input/tank pressure
    'pressure_outlet': ['P3', 'P4'],         # Load pressure
    'flow_rate': ['FJ', 'L2'],              # Main + secondary flow
    'temperature': ['T1', 'T2'],            # Coolant inlet/outlet
    'pressure_drop': 'P3 - P4',             # Computed: ΔP across valve
    'vibration_proxy': 'diff(P1)',          # Pressure ripple = pump vane chatter
}

# COMPONENT SENSORS (Internal: equipment-specific)
component_sensors = {
    'pump': {
        'speed_rpm': 'EMS',                 # Pump speed (if variable displacement)
        'displacement': 'VS',                # Unload valve = pump swashplate proxy
        'case_pressure': 'P2',              # Pump case drain
    },
    'actuator': {
        'position': 'FJ',                    # Rod position or spool spool
        'load_pressure': 'TS',              # Actuator load pressure
        'flow': 'FJ',                       # Duplicated: use for validation
    },
    'accumulator': {
        'stored_pressure': 'TS2',           # Precharge reference
    },
    'fluid': {
        'viscosity': 'FS',                  # ISO grade proxy
        'condition': 'FS2',                 # Degradation index
    }
}

# EXCLUDE
psi_electrical = 'PSI'  # High noise, unclear meaning
```

#### **1.2 Feature Engineering (Per Edge/Node)**

**For EDGES (hydraulic lines):**

```python
def extract_edge_features(pressure_in, pressure_out, flow, temp, window_size=60):
    """
    Extract 48-116D features per edge (line).
    Args: 60-sample window at 10Hz = 6 seconds of history
    """
    features = {}
    
    # === STATIC FEATURES (8D) ===
    features['p_in_current'] = pressure_in[-1]              # [bar]
    features['p_out_current'] = pressure_out[-1]            # [bar]
    features['dp_current'] = pressure_in[-1] - pressure_out[-1]  # [bar]
    features['flow_current'] = flow[-1]                     # [L/min]
    features['temp_current'] = temp[-1]                     # [°C]
    features['p_in_std'] = np.std(pressure_in)              # Ripple
    features['dp_std'] = np.std(np.diff(pressure_out))      # Transient
    features['flow_variance'] = np.var(flow)                # Stability
    
    # === DYNAMIC FEATURES (6D) ===
    features['p_in_trend'] = (pressure_in[-1] - pressure_in[0]) / window_size  # [bar/s]
    features['p_out_trend'] = (pressure_out[-1] - pressure_out[0]) / window_size
    features['dp_trend'] = features['dp_current'] - np.mean(np.diff(pressure_out[-10:]))
    features['flow_trend'] = (flow[-1] - flow[0]) / window_size
    features['temp_trend'] = (temp[-1] - temp[0]) / window_size  # [°C/s]
    features['p_ratio'] = (pressure_in[-1] / (pressure_out[-1] + 1e-6))  # Efficiency proxy
    
    # === TIME-SERIES FEATURES (34D per sensor, up to 3 sensors = 102D) ===
    # For each of 3 sensors: [pressure_in, pressure_out, flow]
    for sensor_idx, sensor_data in enumerate([pressure_in, pressure_out, flow]):
        # Downsample to 34 points (6 seconds → 10 points/sec)
        downsampled = np.interp(
            np.linspace(0, len(sensor_data)-1, 34),
            np.arange(len(sensor_data)),
            sensor_data
        )
        features[f'timeseries_sensor{sensor_idx}'] = downsampled  # [34D]
    
    # === FINAL SHAPE ===
    return np.concatenate([
        np.array([features[k] for k in sorted(features.keys()) if not 'timeseries' in k]),
        features['timeseries_sensor0'],
        features['timeseries_sensor1'],
        features['timeseries_sensor2'],
    ])  # Shape: [14 + 34 + 34 + 34] = [116D] if all three sensors included
           # or [14 + 34] = [48D] if pressure only (as in EDGE_CENTRIC_MIGRATION.md)
```

**For NODES (components):**

```python
def extract_node_features(rpm, position, temp, viscosity, window_size=60):
    """
    Extract 29D features per component.
    """
    features = {}
    
    # === OPERATIONAL STATE (4D) ===
    features['rpm_norm'] = rpm[-1] / 3000  # Normalized to max 3000 RPM
    features['position_norm'] = position[-1] / 100  # Normalized to 0-100%
    features['temp_norm'] = (temp[-1] - 20) / 60  # Normalized to 20-80°C range
    features['viscosity_idx'] = viscosity[-1]  # ISO grade (25 = ISO VG 32, etc.)
    
    # === COMPONENT TYPE ONE-HOT (25D) ===
    # Example: pump = [1, 0, 0, 0, ...]
    component_types = [
        'pump_piston', 'pump_gear', 'pump_vane',  # 3
        'valve_directional', 'valve_proportional', 'valve_relief', 'valve_throttle',  # 4
        'actuator_cylinder', 'actuator_motor',  # 2
        'filter_suction', 'filter_pressure', 'filter_return',  # 3
        'accumulator_bladder', 'accumulator_piston',  # 2
        'cooler_air', 'cooler_water',  # 2
        'motor_electric', 'motor_hydraulic',  # 2
        'sensor_pressure', 'sensor_temp', 'sensor_flow',  # 3
        'tank', 'pump_inlet',  # 2
    ]  # Total: 25 types
    
    one_hot = np.zeros(25)
    # Infer type from known component metadata
    # (This comes from topology.json, not from sensors)
    # one_hot[component_type_idx] = 1
    
    # === FINAL SHAPE ===
    return np.concatenate([
        np.array([features[k] for k in ['rpm_norm', 'position_norm', 'temp_norm', 'viscosity_idx']]),
        one_hot
    ])  # Shape: [4 + 25] = [29D]
```

### **Phase 2: Graph Construction (HybridInferenceRequest → PyG Data)**

#### **2.1 Node Index Mapping**

```python
# From topology.json (example: simple pump-valve-cylinder system)
topology = {
    'components': {
        '0': {'id': 'pump_main', 'type': 'pump_piston'},
        '1': {'id': 'valve_4_3', 'type': 'valve_directional'},
        '2': {'id': 'cylinder_boom', 'type': 'actuator_cylinder'},
        '3': {'id': 'accumulator', 'type': 'accumulator_bladder'},
        '4': {'id': 'tank', 'type': 'tank'},
    },
    'edges': {
        '0-1': {'source': '0', 'target': '1', 'material': 'hose_DN12'},
        '1-2': {'source': '1', 'target': '2', 'material': 'hose_DN10'},
        '2-4': {'source': '2', 'target': '4', 'material': 'hose_DN16'},
        # ... more edges
    }
}

# PyG Data structure:
data = {
    'x': torch.randn(5, 29),  # 5 component nodes, 29D features
    'edge_index': torch.tensor([
        [0, 1, 2],  # source nodes
        [1, 2, 4],  # target nodes
    ]),  # 3 edges
    'edge_attr': torch.randn(3, 48),  # 3 edges, 48D features (minimum)
}
```

#### **2.2 Batch Assembly (DataLoader)**

```python
class HydraulicCycleDataset(Dataset):
    """Load N cycles from disk and create mini-batches."""
    
    def __init__(self, cycles_dir: str, topology: dict, window_size: int = 60):
        self.cycles = sorted(Path(cycles_dir).glob('cycle_*.csv'))
        self.topology = topology
        self.window_size = window_size
    
    def __len__(self):
        return len(self.cycles)
    
    def __getitem__(self, idx):
        # Load one cycle
        cycle_data = pd.read_csv(self.cycles[idx])  # Shape: [650, 16]
        
        # Extract window (e.g., rows 100:160 = 6-second window)
        window = cycle_data.iloc[100:160]
        
        # Build edge features from window
        edge_features = extract_edge_features(
            pressure_in=window['P1'].values,
            pressure_out=window['P3'].values,
            flow=window['FJ'].values,
            temp=window['T1'].values,
            window_size=self.window_size
        )  # Shape: [48D] or [116D]
        
        # Build node features from window
        node_features = extract_node_features(
            rpm=window['EMS'].values,
            position=window['FJ'].values,
            temp=window['TS'].values,
            viscosity=window['FS'].values,
            window_size=self.window_size
        )  # Shape: [29D]
        
        # Return PyG Data object
        return Data(
            x=torch.from_numpy(np.repeat(node_features[np.newaxis, :], 5, axis=0)).float(),
            edge_index=torch.tensor(self.topology['edge_index']),
            edge_attr=torch.from_numpy(edge_features[np.newaxis, :]).float(),
            y_health=torch.tensor([1.0]),  # Example label: healthy=1.0
        )

# Usage:
dataset = HydraulicCycleDataset('data/cycles', topology)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 🎯 **Model Quality Expectations**

### **What Can We Realistically Achieve?**

Based on:
- 16 sensor channels, 10 Hz sampling
- Real hydraulic cycles (~65 seconds each)
- Phase 3.2 edge-centric graph architecture
- Homogeneous GNN (GAT/GCN) baseline

#### **Task 1: Line-Level Anomaly Detection (Easy)**

**Problem:** Detect pressure/flow anomalies on specific hydraulic lines

**What the data shows:**
- ✅ Very clear patterns: P1 spikes, FJ ramps, T1 drifts
- ✅ Multiple operational phases (startup, steady, cool-down)
- ✅ Low noise on slow signals (T1, T2, TS)
- ⚠️ High-frequency noise on P1 (pump ripple)

**Expected Performance (Phase 3.2):**
```
ANOMALY DETECTION (Binary: Normal / Anomalous Flow)
├─ Precision: 0.90-0.95        (few false alarms)
├─ Recall: 0.85-0.92          (catch most anomalies)
├─ F1 Score: 0.88-0.93        
├─ AUC: 0.93-0.97
└─ Per-line Accuracy: 85-92%

LEAK LOCALIZATION (Which line is leaking?)
├─ Precision: 0.80-0.90        (when we say "line 5 leaks", we're right)
├─ Top-1 Accuracy: 88%         (correct line in top prediction)
├─ Top-3 Accuracy: 96%         (correct line in top 3)
└─ False Positive Rate: <5%
```

**Why good?**
- Sensors directly on edges (lines)
- 116D feature space per line (rich representation)
- Clear signal changes when pressure/flow anomaly occurs

**Failure modes:**
- ❌ Proportional valve hysteresis (P1 oscillates naturally)
- ❌ Temperature compensation (T affects viscosity → affects pressure drop)
- ❌ Accumulator pulsations (TS2 noise)

---

#### **Task 2: Component Health Classification (Medium)**

**Problem:** Predict component state (healthy / degraded / failed)

**What the data shows:**
- ✅ Slow temperature trends (T1, T2) = system thermal stress
- ✅ Pump displacement changes (VS drift) = pump wear
- ✅ Flow asymmetries (FJ vs L2) = seal leakage
- ⚠️ But mostly "healthy" operation (class imbalance)
- ⚠️ No explicit failure labels in single cycle

**Expected Performance (Phase 3.2 homogeneous):**
```
COMPONENT HEALTH (Multi-class: Healthy / Degraded / Failed)
├─ Healthy samples: 0.92 F1    (easy: most of data is healthy)
├─ Degraded samples: 0.70 F1   (harder: subtle signature)
├─ Failed samples: 0.80 F1     (medium: clear symptoms)
├─ Weighted Average F1: 0.85
└─ Macro Average F1: 0.81

COMPONENT TYPE PREDICTION (Which component is this?)
├─ Accuracy: 0.88              (identify pump vs valve vs cylinder)
└─ Confusion: pump↔valve (10% misclassification)
```

**Why medium?**
- Component labels must come from separate ground truth (not visible in one cycle)
- Need multiple cycles/equipment to learn component signatures
- Message passing through graph helps: valve temp ← pump heat ← bearing wear

**Failure modes:**
- ❌ Single cycle insufficient for aging/degradation (need 100+ cycles per equipment)
- ❌ Component-level signals sparse (only TS, TS2, TS3, EMS)
- ❌ Topology unknown (edge count, connections)

---

#### **Task 3: Remaining Useful Life (RUL) Prediction (Hard)**

**Problem:** Predict days/hours until equipment failure

**What the data shows:**
- ✅ Temperature trends (T1, T2 drift up ~0.3°C over 65s)
- ✅ Pressure ripple growth (P1 std increases with wear)
- ⚠️ Single cycle = single timestamp → hard to extrapolate
- ❌ No failure time labeled (when does this equipment fail?)

**Expected Performance (Phase 3.2):**
```
RUL PREDICTION (Days until failure)
├─ Mean Absolute Error (MAE): ±8-15 days     (±12% error)
├─ RMSE: ±12-20 days
├─ Asymmetric Loss (penalizes over-optimism): 0.15-0.25
├─ Horizon Accuracy:
│   ├─ 24h ahead: 75% (good)
│   ├─ 72h ahead: 65% (medium)
│   └─ 168h (7d) ahead: 50% (random guessing)
└─ Correlation with True RUL: r² = 0.55-0.70
```

**Why hard?**
- RUL is a regression task (not binary)
- Need time-series correlation (current temp → future failure)
- Extrapolation outside training domain risky
- Equipment degradation is nonlinear

**Failure modes:**
- ❌ Not enough cycles from same equipment before failure
- ❌ Temperature plateau (no additional signal after 40s)
- ❌ Wear rate unknown (depends on load history, not visible in one cycle)

---

### **Phase 3.3 Expected Improvement (Hetero Incidence Graph)**

After switching to **component-centric heterogeneous graph:**

```
PHASE 3.2 (Homogeneous, Edge-Centric)
  Line Anomaly F1: 0.91 ✓
  Component Health F1: 0.85
  RUL MAE: ±10 days
           ↓
PHASE 3.3 (Heterogeneous, Component-Centric)
  Component Health F1: 0.94 (+10%)
  Multi-label Accuracy: 0.87 (+2%)
  RUL MAE: ±8 days (-20%)
  
Why better?
  ✅ Explicit component nodes (main prediction target)
  ✅ Multi-label per component (pump: cavitation + internal_leak)
  ✅ Bipartite message passing (line signals → component state)
  ✅ Better interpretability (attention: which lines caused pump fail?)
```

---

## 📝 **Practical Data Preparation Checklist**

### **Step 1: Dataset Assembly**

```
✅ Collect cycles:
   ├─ 500-1000 healthy cycles (baseline)
   ├─ 200-500 degraded cycles (wear, contamination)
   ├─ 50-200 failure cycles (catastrophic failure)
   └─ Total: 1000-1500 cycles minimum

✅ Annotate with ground truth:
   ├─ Component type (pump, valve, cylinder, ...)
   ├─ Health state (healthy, degraded, failed)
   ├─ Failure type if applicable (cavitation, seal leak, ...)
   └─ Time to failure if available (e.g., "failed 10 days after this cycle")

✅ Sensor calibration:
   ├─ Pressure sensors: 0-400 bar, ±1% accuracy
   ├─ Temperature sensors: -10 to +100°C, ±0.5°C accuracy
   ├─ Flow meters: 0-200 L/min, ±2% accuracy
   └─ Electrical signals: match DAQ specs
```

### **Step 2: Feature Engineering**

```
✅ Handle missing data:
   ├─ Linear interpolation for <5% gaps
   ├─ Forward fill for isolated points
   └─ Drop cycles with >10% missing data

✅ Normalize/scale:
   ├─ Pressure: (x - 100) / 200 → [-0.5, 0.5]
   ├─ Temperature: (x - 35) / 30 → [-1, 1]
   ├─ Flow: x / 200 → [0, 1]
   └─ RPM: x / 3000 → [0, 1]

✅ Outlier detection:
   ├─ Remove values >3σ from mean (likely sensor fault)
   ├─ Check for data clipping (sudden flat regions)
   └─ Validate pressure drops are positive

✅ Temporal alignment:
   ├─ Resample to 10 Hz (if different)
   ├─ Align to cycle start (e.g., P1 spike = t=0)
   └─ Crop to fixed length (e.g., 650 samples = 65s)
```

### **Step 3: Train/Val/Test Split**

```
✅ Stratified split:
   ├─ Training: 70% (1000 cycles)
   │  ├─ Healthy: 700 cycles
   │  ├─ Degraded: 250 cycles
   │  └─ Failed: 50 cycles
   ├─ Validation: 15% (215 cycles)
   └─ Test: 15% (215 cycles)

✅ No data leakage:
   ├─ Split by equipment ID (all cycles from same machine → one set)
   ├─ No chronological overlap (if testing on time-forward)
   └─ Independent of cycle order

✅ Class balance:
   ├─ If imbalanced, use weighted loss:
   │  └─ weight = total / (num_classes × class_count)
   ├─ Or oversample failed cycles (2-3x)
   └─ Or use focal loss (penalizes easy negatives)
```

### **Step 4: Validation & Sanity Checks**

```
✅ Sanity checks:
   ├─ P1 > P4 (pressure monotonic in main line)
   ├─ T1, T2 > 0 (temperature physical)
   ├─ FJ, L2 >= 0 (flow non-negative)
   ├─ VS ∈ [0, 100] (valve position bounded)
   └─ Sum(flow out) ≈ Sum(flow in) (conservation)

✅ Feature correlation:
   ├─ High correlation (>0.9) between T1 and T2 (normal)
   ├─ No perfect multicollinearity (VIF < 10)
   ├─ Outliers don't dominate (check distribution plots)
   └─ Statistical summary: mean, std, min, max, median

✅ Temporal consistency:
   ├─ No sudden jumps in P1, T1, FJ (check derivatives)
   ├─ Monotonic temperature increase (T1, T2 trends)
   └─ Valve position changes smooth (VS not jittery)
```

---

## 📖 **Recommended Training Recipe**

### **Week 1-2: Baseline (Phase 3.2 Homogeneous)**

```python
from src.training import HydraulicGNNModule, create_development_trainer

# Minimal model
module = HydraulicGNNModule(
    in_channels=48,              # Basic edge features (no time-series)
    hidden_channels=64,          # Lightweight
    num_heads=4,                 # Single-head attention
    num_gat_layers=2,            # Shallow graph
    learning_rate=0.001,
    loss_weighting="fixed",
    loss_weights={
        "line_anomaly": 1.0,     # Primary task
    }
)

trainer = create_development_trainer()  # 50 epochs, fast iteration
trainer.fit(module, train_loader, val_loader)

# Expected result: ~0.88 F1 on line anomaly after 20-30 epochs
```

### **Week 3-4: Production Model (Phase 3.2 Edge-Centric)**

```python
# Rich feature model
module = HydraulicGNNModule(
    in_channels=116,             # Rich edge features (with time-series)
    hidden_channels=128,         # Medium capacity
    num_heads=8,                 # Multi-head
    num_gat_layers=3,            # Deeper
    lstm_hidden=256,
    lstm_layers=2,
    learning_rate=0.0005,        # Reduced for stability
    loss_weighting="uncertainty",  # Adaptive loss balancing
    loss_weights={
        "line_anomaly": 1.0,
        "component_health": 0.5,
        "graph_rul": 0.3,
    }
)

trainer = create_production_trainer(max_epochs=200)
trainer.fit(module, train_loader, val_loader)

# Expected result:
#   Line Anomaly F1: 0.91-0.93
#   Component Health F1: 0.83-0.87
#   RUL MAE: ±9-12 days
```

### **Week 5-6: Phase 3.3 Heterogeneous**

```python
from src.training import HeteroGNNModule

# Same input (HybridInferenceRequest)
# Different graph (HeteroData) and model (HeteroConv)

module = HeteroGNNModule(
    component_in_dim=29,
    line_in_dim=116,
    embedding_dim=128,
    num_layers=3,
    heads=4,
    learning_rate=0.0005,
    loss_weights={
        "component_multi_label": 1.0,  # Primary
        "line_anomaly_aux": 0.2,       # Auxiliary
    }
)

trainer = create_production_trainer(max_epochs=200)
trainer.fit(module, hetero_train_loader, hetero_val_loader)

# Expected result:
#   Component Fault F1: 0.94-0.96
#   Multi-label Accuracy: 0.86-0.89
#   RUL MAE: ±7-10 days
```

---

## 📊 **Monitoring & Debugging**

### **What to Watch During Training**

```python
# TensorBoard monitoring
monitored_metrics = [
    'train/total_loss',        # Should decrease smoothly
    'val/total_loss',          # Should decrease, then plateau
    'val/line_anomaly_f1',     # Should reach 0.88+ by epoch 30
    'val/component_health_f1', # Should reach 0.80+ by epoch 50
    'val/rul_mae',             # Should reach <12 days by epoch 100
]

# Red flags
red_flags = {
    'loss explodes': "Check learning rate, gradient clipping",
    'validation diverges': "Overfitting: increase dropout/weight_decay",
    'F1 plateaus early': "Model capacity too low or learning rate too high",
    'RUL doesn\'t improve': "Not enough temporal context or label noise",
}
```

---

## 🚀 **Next Steps**

1. ✅ Assemble 1000+ labeled cycles
2. ✅ Run feature engineering pipeline
3. ✅ Train Phase 3.2 baseline (week 1-2)
4. ✅ Production tuning (week 3-4)
5. ✅ Implement Phase 3.3 (week 5-6)
6. ✅ A/B test and select best model
7. ✅ Deploy to production

---

**Estimated Timeline:** 6-8 weeks from raw data to production model  
**Data Requirements:** 1000-2000 labeled cycles minimum  
**Expected Quality:** Phase 3.2 = 88-92% line detection; Phase 3.3 = 94-96% component prediction  
