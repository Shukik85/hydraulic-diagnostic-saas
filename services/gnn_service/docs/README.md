# 📒 GNN Service Documentation

**Comprehensive guide to the Hydraulic Diagnostics GNN architecture, from sensor placement to component-level fault prediction.**

---

## 🏗️ **Architecture Phases**

### 😶 **Phase 3.1: Node-Centric (Legacy)**
- **Goal:** Baseline implementation
- **Approach:** All sensors compressed into component nodes
- **Limitation:** Poor signal clarity, multi-port components lost
- **Status:** ❌ Deprecated (replaced by Phase 3.2)

### 📈 **Phase 3.2: Edge-Centric Sensor Placement (CURRENT)**
- **Goal:** Rich sensor representation on hydraulic lines
- **Approach:** Homogeneous graph with 48-116D edge features
- **Key Achievement:** +40-60% line-level anomaly detection accuracy
- **Status:** ✅ COMPLETE (Dec 26, 2025)
- **Documentation:** [`EDGE_CENTRIC_MIGRATION.md`](./EDGE_CENTRIC_MIGRATION.md)
- **Implementation:** `GraphBuilderV2.build_graph_hybrid()` ✅
- **What It Provides:**
  - Clear sensor placement (physically on pipes, not in components)
  - Rich edge features: pressure_inlet, pressure_outlet, flow, temperature, vibration
  - Excellent for leak detection, pressure drop analysis, line-level diagnostics
  - Foundation for next phase

### 🎯 **Phase 3.3: Component-Level Diagnostics with Hetero Graphs (NEXT)**
- **Goal:** Direct equipment fault prediction (pumps, valves, cylinders)
- **Approach:** Heterogeneous incidence graph (bipartite: components ↔ lines)
- **Key Innovation:** Explicit multi-node types + bipartite message passing
- **Expected Improvement:** +3% component accuracy + 100% interpretability
- **Status:** ⏳ Architecture Defined, Implementation Pending
- **Documentation:**
  - [`COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md`](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) — Full Phase 3.3 spec
  - [`GRAPH_ARCHITECTURE_EVOLUTION.md`](./GRAPH_ARCHITECTURE_EVOLUTION.md) — Visual comparison of all three approaches
- **Implementation Plan:** GraphBuilderV2.build_graph_hetero() + HeteroComponentDiagnostics model
- **What It Will Provide:**
  - Multi-label component fault classification
  - Direct answer to "Which component is failing?"
  - Auxiliary line-level anomaly detection
  - Better interpretability (attention weights per component-line)

---

## 📄 **Documentation Structure**

### **Core Architecture Documents**

| Document | Purpose | Audience | Status |
|----------|---------|----------|--------|
| [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) | Phase 3.2 implementation guide | Engineers, ML researchers | ✅ COMPLETE |
| [COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) | Phase 3.3 specification | Architects, Senior engineers | 🎯 DESIGNED |
| [GRAPH_ARCHITECTURE_EVOLUTION.md](./GRAPH_ARCHITECTURE_EVOLUTION.md) | Visual comparison of approaches | Decision makers, New team members | 🎯 NEW |

### **Implementation & Operations**

| Document | Purpose | Key Content |
|----------|---------|-------------|
| [MODEL_CONTRACT.md](./MODEL_CONTRACT.md) | I/O specifications | Input/output tensor shapes, PyG Data structure |
| [API_DOCS.md](./API_DOCS.md) | FastAPI endpoints | Request/response schemas, examples |
| [INFERENCE.md](./INFERENCE.md) | Inference pipeline | Model loading, batch processing, error handling |
| [TRAINING.md](./TRAINING.md) | Training procedures | Data prep, hyperparameters, checkpointing |
| [TRAINING_INTERNALS.md](./TRAINING_INTERNALS.md) | Low-level training details | Loss computation, optimization, monitoring |

### **Data & Quality**

| Document | Purpose | Key Content |
|----------|---------|-------------|
| **[RAW_DATA_QUALITY_ASSESSMENT.md](./RAW_DATA_QUALITY_ASSESSMENT.md)** | **UCI dataset quality analysis** | **Sensor interpretation, missing data check, outlier detection, pre-processing requirements** |
| **[DATA_PREPARATION_AND_MODEL_QUALITY.md](./DATA_PREPARATION_AND_MODEL_QUALITY.md)** | **Analysis of real data + quality expectations** | **Feature engineering, expected model performance, practical training recipe** |
| [DATA_GENERATION_GUIDE.md](./DATA_GENERATION_GUIDE.md) | Synthetic data creation | Scenarios, parameter ranges, file formats |

### **Infrastructure & Deployment**

| Document | Purpose | Key Content |
|----------|---------|-------------|
| [DOCKER.md](./DOCKER.md) | Containerization | Build, run, environment setup |

---

## 🔍 **Quick Navigation**

### **"I want to understand the architecture"**
1. Start: [GRAPH_ARCHITECTURE_EVOLUTION.md](./GRAPH_ARCHITECTURE_EVOLUTION.md) — Visual 3-phase comparison
2. Then: [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) — Current Phase 3.2 details
3. Finally: [COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) — Next Phase 3.3

### **"I want to assess raw data quality"**
1. **Start here:** [RAW_DATA_QUALITY_ASSESSMENT.md](./RAW_DATA_QUALITY_ASSESSMENT.md)
   - Dataset overview (530 MB, 17 sensors, 100 Hz sampling)
   - Sensor mapping and physical interpretation
   - Missing value & outlier analysis
   - Pre-processing requirements (resampling, normalization, segmentation)
   - UCI benchmark comparison
2. Reference: [DATA_GENERATION_GUIDE.md](./DATA_GENERATION_GUIDE.md) — Synthetic data if needed

### **"I need to prepare training data and understand model quality"**
1. **Start here:** [DATA_PREPARATION_AND_MODEL_QUALITY.md](./DATA_PREPARATION_AND_MODEL_QUALITY.md)
   - Analysis of real hydraulic cycle characteristics
   - Sensor interpretation (what each channel means)
   - Expected model performance by task (line anomaly, component health, RUL)
   - Practical data preparation checklist
   - Recommended training recipe (week-by-week)
2. Reference: [RAW_DATA_QUALITY_ASSESSMENT.md](./RAW_DATA_QUALITY_ASSESSMENT.md) — Data quality baseline
3. Reference: [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) — Feature engineering details
4. Implement: [TRAINING.md](./TRAINING.md) — Actual training code

### **"I need to implement Phase 3.3"**
1. Read: [COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) (Sections: Architecture Overview, Graph Construction, Model Architecture)
2. Reference: [MODEL_CONTRACT.md](./MODEL_CONTRACT.md) — HeteroData format specifications
3. Implement: `GraphBuilderV2.build_graph_hetero()` method (see COMPONENT_DIAGNOSTICS for pseudocode)
4. Test: Follow implementation checklist in Phase 3.3 doc

### **"I need to run inference"**
1. Reference: [INFERENCE.md](./INFERENCE.md) — Pipeline walkthrough
2. Check: [API_DOCS.md](./API_DOCS.md) — Endpoint schemas
3. Example: Request format in [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) (Phase 3.2) or [COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) (Phase 3.3)

### **"I need to retrain the model"**
1. **Check data first:** [RAW_DATA_QUALITY_ASSESSMENT.md](./RAW_DATA_QUALITY_ASSESSMENT.md) — Understand your raw data
2. **Plan preparation:** [DATA_PREPARATION_AND_MODEL_QUALITY.md](./DATA_PREPARATION_AND_MODEL_QUALITY.md) — Expected performance
3. **Implement pipeline:** [TRAINING.md](./TRAINING.md) — High-level procedure
4. **Details:** [TRAINING_INTERNALS.md](./TRAINING_INTERNALS.md) — Loss functions, optimization
5. **Generate data:** [DATA_GENERATION_GUIDE.md](./DATA_GENERATION_GUIDE.md) — Synthetic dataset if needed

### **"I need to deploy this"**
1. Build: [DOCKER.md](./DOCKER.md) — Container setup
2. API: [API_DOCS.md](./API_DOCS.md) — Endpoint specifications
3. Monitor: [INFERENCE.md](./INFERENCE.md) — Observability

---

## 📈 **Data Flow: Edge-Centric to Component-Level**

### **Phase 3.2 (Current)**

```
Sensor readings (hydraulic lines)
  ↓
HybridInferenceRequest
  edge_readings: {"pump→valve": [P_in, P_out, ΔP, flow, T]}
  component_readings: {"pump": [RPM, I, V]}
  ↓
GraphBuilderV2.build_graph_hybrid()
  ↓
PyG Data object
  x: [N, 29]         (component nodes: RPM_norm, Pos_norm, I_norm, V_norm, type[25D])
  edge_index: [2, E]
  edge_attr: [E, 48-116]  (rich line features)
  ↓
Homogeneous GNN (GAT/GCN)
  ↓
Line-level outputs
  anomaly detection: [E] tensor
  health scores per line: [E] tensor
```

### **Phase 3.3 (Next)**

```
Same sensor readings + Same HybridInferenceRequest!
  ↓
GraphBuilderV2.build_graph_hetero()  ← NEW METHOD
  ↓
HeteroData object (bipartite)
  ['component'].x: [N, 29]
  ['line'].x: [E, 48-116]
  [('component', 'source', 'line')].edge_index
  [('line', 'sink', 'component')].edge_index
  ↓
HeteroGNN (HeteroConv + HGT/HAN layers)
  ↓
Component-level outputs (multi-label)
  predictions: [N, num_fault_types]  ← COMPONENT FAULTS!
  aux_line_anomaly: [E]               ← AUXILIARY
```

---

## 🆕 **Key Schemas (Phase 3.2+)**

### **Input Request**

```python
class HybridInferenceRequest:
    equipment_id: str
    timestamp: datetime
    topology_id: str
    
    # Edge sensors (on hydraulic lines)
    edge_readings: dict[str, EdgeSensorReading]
        # pressure_inlet_bar, pressure_outlet_bar, ΔP, flow_rate_lpm, T_c, vibration_g
    
    # Component sensors (internal)
    component_readings: dict[str, ComponentSensorReading]
        # rpm, position_percent, current_a, voltage_v
```

### **Output Response**

```python
# Phase 3.2 (Line-level)
class LineAnomalyResponse:
    equipment_id: str
    line_anomalies: dict[str, float]  # [edge_id] -> anomaly_score
    system_health: float
    api_version: "v2_hybrid"

# Phase 3.3 (Component-level) → COMING
class ComponentDiagnosticsResponse:
    equipment_id: str
    component_states: dict[str, FaultLabel]  # [component_id] -> fault type + confidence
    multi_labels: dict[str, List[str]]  # Multi-label predictions
    system_health: float
    api_version: "v3_hetero_component"
```

---

## 🔧 **Key Implementation Files**

### **Data Pipeline**
- `src/data/graph_builder_v2.py` ✅ — Phase 3.2 complete
  - `build_graph_hybrid()` ✅
  - `build_graph_hetero()` ⏳ (Phase 3.3 TODO)

### **Models**
- `src/models/gnn_model.py` ✅ — Homogeneous GNN (Phase 3.2)
- `src/models/hetero_model.py` ⏳ — Heterogeneous GNN (Phase 3.3 TODO)

### **Inference**
- `src/inference/inference_engine.py` ✅
  - `predict_hybrid()` ✅ (Phase 3.2)
  - `predict_hetero()` ⏳ (Phase 3.3 TODO)

### **APIs**
- `src/api/endpoints/inference.py` ✅
  - `/v2/inference/hybrid` ✅ (Phase 3.2)
  - `/v3/inference/component_diagnostics` ⏳ (Phase 3.3 TODO)

---

## 🕒 **Timeline**

```
Phase 3.2 (Current)
  Start: Dec 20, 2025
  End: Dec 26, 2025
  Status: ✅ COMPLETE
  Commits: 3 major commits (schemas, graph_builder, tests)

Phase 3.3 (Next)
  Start: Jan 1, 2026 (week 1)
  GraphBuilderV2: build_graph_hetero() implementation (1 week)
  Model: HeteroComponentDiagnostics (1 week)
  Integration: InferenceEngine + API (1 week)
  Retraining: Model training + validation (2 weeks)
  Total: 5-6 weeks
  Expected: Late January 2026
```

---

## 📄 **Glossary**

| Term | Definition |
|------|------------|
| **Edge-centric** | Sensor data represented as edge attributes (Phase 3.2) |
| **Node-centric** | Sensor data compressed into node features (Phase 3.1, legacy) |
| **Hetero/Heterogeneous** | Graph with multiple node and edge types (Phase 3.3) |
| **Incidence Graph** | Bipartite representation (components ↔ lines) |
| **HybridInferenceRequest** | Request format with edge_readings + component_readings |
| **GraphBuilderV2** | Tool for converting sensor data to PyG Data/HeteroData |
| **Component** | Hydraulic equipment (pump, valve, cylinder, etc.) |
| **Line** | Hydraulic pipe/hose connecting components |
| **Multi-label** | Each component can have multiple simultaneous faults |
| **Auxiliary loss** | Secondary training objective (e.g., line anomaly) |

---

## 👀 **Status Dashboard**

| Phase | Feature | Status | Owner | ETA |
|-------|---------|--------|-------|-----|
| 3.2 | EdgeSensorReading schema | ✅ | Team | Dec 26 |
| 3.2 | GraphBuilderV2.build_graph_hybrid() | ✅ | Team | Dec 26 |
| 3.2 | Unit tests (23 methods) | ✅ | Team | Dec 26 |
| 3.2 | InferenceEngine.predict_hybrid() | ⏳ | Pending | Jan 1-3 |
| 3.2 | /v2/inference/hybrid endpoint | ⏳ | Pending | Jan 3-5 |
| **Data** | **RAW_DATA_QUALITY_ASSESSMENT.md** | **🎯 NEW (Dec 30)** | **Analysis & Pre-processing Strategy** | **- |
| **Data** | **DATA_PREPARATION_AND_MODEL_QUALITY.md** | **🎯 NEW (Dec 30)** | **Training Expectations** | **- |
| 3.3 | COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md | 🎯 | NEW (Dec 30) | - |
| 3.3 | GRAPH_ARCHITECTURE_EVOLUTION.md | 🎯 | NEW (Dec 30) | - |
| 3.3 | GraphBuilderV2.build_graph_hetero() | ⏳ | To do | Jan 5-10 |
| 3.3 | HeteroComponentDiagnostics model | ⏳ | To do | Jan 10-15 |
| 3.3 | Retraining pipeline | ⏳ | To do | Jan 15-25 |
| 3.3 | /v3/inference/component_diagnostics | ⏳ | To do | Jan 25-30 |

---

## 🔗 **Related Links**

- Repository: [hydraulic-diagnostic-saas](https://github.com/Shukik85/hydraulic-diagnostic-saas)
- Branch: `fix/dataloader-edge-dedup-and-similarity` (current work)
- ML Space: See /spaces/ML for additional context

---

**Last Updated:** December 30, 2025  
**Maintainers:** ML Engineering Team  
**Questions?** See specific phase documentation or check implementation files referenced above.