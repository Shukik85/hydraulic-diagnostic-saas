# 📈 Graph Architecture Evolution

**From Simple Homogeneous Graphs to Heterogeneous Incidence Graphs**

---

## 🏗️ **Three Architectural Approaches**

### **Approach 1: Homogeneous Node-Centric (Early Phase)**

```
┌─────────────────────────────────────────────────────┐
│ Graph Type:  Homogeneous (all nodes same type)      │
│ Primary Use: Baseline / proof-of-concept            │
│ Limitation:  Poor signal-to-noise at node level     │
└─────────────────────────────────────────────────────┘

        Pump(150, 65, 115, 1450)  ─[e1]─  Valve(148, 64, 110)
        │
        │ Features: [P_bar, T_c, Flow_lpm, RPM, Pos%, ...]
        │ ~29D per node
        │
        └─ Problem: Where does Pressure 150 come from?
           "Pump node" conflates internal sensors (RPM=1450)
           with output signals (P=150 at pump outlet).
           Valve receives edge[e1] data anyway, so node features
           are redundant and confusing.
```

**Issues:**
- ❌ Sensors "assigned" to nodes ambiguously
- ❌ Multi-port components (pump: inlet, discharge) → one node → lost information
- ❌ Can't distinguish "pump internal pressure" vs "pump outlet pressure"
- ❌ Edge features exist but aren't the primary representation

---

### **Approach 2: Homogeneous Edge-Centric + Line-Graph (Phase 3.2)**

```
┌─────────────────────────────────────────────────────┐
│ Graph Type 2a: Original Homogeneous                 │
│ Primary Use:  Direct sensor mapping                 │
│ Advantage:    Sensors clearly on edges              │
└─────────────────────────────────────────────────────┘

Pump ─[P=150, T=65, F=115, ΔP=-]─ Valve ─[P=148, T=64, F=110, ΔP=2]─ Cylinder
│                                                      │
└─ RPM=1450, I=25A, V=400V                           └─ Pos=45%


┌─────────────────────────────────────────────────────┐
│ Graph Type 2b: Line-Graph (Edges → Nodes)           │
│ Primary Use:  Edge-level anomaly detection          │
│ Advantage:    Lines are explicit entities           │
└─────────────────────────────────────────────────────┘

Components:      Pump ─────── Valve ─────── Cylinder
                  │             │             │
Line nodes:   [edge_1]      [edge_2]      [edge_3]
(as actual nodes) │             │             │
Edge index:   Connected via component adjacency
              Pump↔Valve, Valve↔Cylinder

Features:
  - Pump node: [RPM_norm, I_norm, V_norm, 25D_type]
  - edge_1 node: [P_in, P_out, ΔP, F, T, V_vibr, ...48D...]
  - edge_2 node: [P_in, P_out, ΔP, F, T, V_vibr, ...48D...]
  - Valve node: [Pos_norm, 29D_type]
  - ...

✅ Advantages:
  - Sensors clearly on "line" nodes
  - Rich edge features (48-116D) now visible as node features
  - Better for line-level anomaly detection (leak, ΔP spike, etc.)

⚠️ Limitations:
  - Components are still nodes, but aren't the primary focus
  - Model learns "which lines are anomalous" naturally
  - But "which component caused this line anomaly?" → indirect inference
  - Multi-port component structure still not explicit
```

**Insights:**
- ✅ Physical reality: sensors ARE on lines (pipes), not in components
- ✅ Rich signal representation via edge_attr (48-116D)
- ⚠️ But for *equipment diagnostics* (fault prediction), components should be primary nodes
- ⚠️ Current architecture makes component-level targets awkward (head on component node, but best signals on edge nodes)

---

### **Approach 3: Heterogeneous Incidence Graph (Phase 3.3) ← PROPOSED**

```
┌──────────────────────────────────────────────────────────┐
│ Graph Type: Heterogeneous Bipartite (Incidence Format)   │
│ Primary Use: Component-level fault prediction            │
│ Advantage:  Explicit roles: equipment vs sensors         │
└──────────────────────────────────────────────────────────┘

    NODE TYPES                    RELATION TYPES
    ─────────────────────────────────────────────
    component (25 types):         component ──[source]──> line
      • Pump                      line ──[sink]──> component
      • Valve                     component ──[self]──> component
      • Cylinder                  (indirect via line)
      • Tank
      • ...

    line (hydro-dynamic):
      • Carries sensor signals
      • Explicit inlet/outlet


    BIPARTITE STRUCTURE (Incidence Graph):

                 [Pump_1]
                    ↓ (source)
              [Line: pump→valve]  ← [RPM, I, V, type] node features
                    ↓              ← [P_in, P_out, ΔP, F, T, V] edge features
                 (sink) ↓
                 [Valve_1]  
                    ↓ (source)  
              [Line: valve→cyl]  ← [P_in, P_out, ΔP, F, T, V]
                    ↓
                 (sink) ↓
               [Cylinder_1]  ← [Pos, type]

    MESSAGE PASSING (simplified):
    
    Round 1:
      Pump_1.x[RPM, I, V] ──→ [Line].x updates (source relation)
      [Line].x ──→ Valve_1.x updates (sink relation)
    
    Round 2:
      Valve_1.x ──→ [Line2].x updates
      [Line2].x ──→ Cyl_1.x updates
    
    Classification Head:
      Pump_1.embedding ──(MLP)──> logits[cavitation, leak, stuck, ...]
      Valve_1.embedding ──(MLP)──> logits[cavitation, leak, stuck, ...]
      Cyl_1.embedding ──(MLP)──> logits[cavitation, leak, stuck, ...]


✅ ADVANTAGES:
  1. **Explicit Multi-Port Handling:**
     - Pump has inlet, discharge, case drain → each gets own line(s)
     - Valve has 4-6 ports → each gets own line
     - Model naturally understands topology
  
  2. **Physical Reality Match:**
     - Component = equipment (tangible)
     - Line = sensor carrier (pipes/hoses)
     - Relations = connections
  
  3. **Cleaner Classification:**
     - Head: component.embedding → logits
     - Evidence: trace back to incident lines
     - "Pump cavitation? Look at pump→valve line ΔP and vibration."
  
  4. **Auxiliary Tasks:**
     - Main: component multi-label classification
     - Auxiliary: line anomaly detection (shared losses)
     - Both reinforce each other
  
  5. **Scalability:**
     - O(c + l + e) where c=components, l=lines, e=edges
     - vs O(c²) for fully connected component graph
  
  6. **Interpretability:**
     - Attention weights: which lines matter most for pump diagnosis?
     - Heat-map: visualize component→line→component signal flow
```

---

## 🔄 **Comparison Table**

| Criterion | Homogeneous Node-Centric | Homogeneous Edge-Centric | Hetero Incidence | 
|-----------|------------------------|----------------------|------------------|
| **Primary entities** | Components | Components + Edges | Components + Lines |
| **Signal placement** | Ambiguous | Clear (on edges) | **Clear (line nodes)** |
| **Multi-port handling** | Poor (one node/component) | Still poor | **Excellent (bipartite)** |
| **Node types** | 1 (all components) | 1 (all components) | **2 (component, line)** |
| **Relation types** | 1 (component→component) | 1 (component→component) | **2 (source, sink) + self** |
| **Main prediction task** | Component health | Line anomaly | **Component fault (primary)** |
| **Auxiliary task** | - | - | **Line anomaly (aux loss)** |
| **Model complexity** | Simple GCN/GAT | GCN/GAT + edge_attr | **HeteroConv (HGT/HAN)** |
| **Inference latency** | ~30ms | ~50ms | ~45ms |
| **Interpretability** | Low (nodes conflate) | Medium (edge features) | **High (explicit roles)** |
| **Expected F1** | 0.70 | 0.91 (lines) | **0.94 (components)** |
| **Physical alignment** | Low | Medium | **High** |

---

## 🎯 **Decision: Why Phase 3.3?**

### **Current State (Phase 3.2 Edge-Centric)**

Works well for:
- ✅ Detecting pressure/flow anomalies on specific lines
- ✅ Leak localization (which line is leaking?)
- ✅ Rich sensor representation (48-116D per line)

But struggles with:
- ❌ "Is the pump cavitating?" → requires indirect inference from multiple lines
- ❌ "Which component caused the vibration spike?" → trace back through edges
- ❌ Multi-label component faults (pump can have cavitation AND internal leak simultaneously)
- ❌ Model isn't optimized for component-level classification (head on component node, but signals on edge nodes)

### **Proposed Solution (Phase 3.3 Hetero Incidence)**

Optimal for:
- ✅ Direct component fault prediction (logits on component node)
- ✅ Multi-label classification per component
- ✅ Auxiliary line-level anomaly detection (optional aux loss)
- ✅ Explainability (attention: which lines → which component faults?)
- ✅ Scales to complex multi-port systems
- ✅ Aligns with hydraulic domain concepts (components ↔ lines)

---

## 📊 **Data Migration Path**

```
Phase 3.2 (Edge-centric):
  HybridInferenceRequest
    ├─ edge_readings["pump→valve"] → EdgeSensorReading
    └─ component_readings["pump"] → ComponentSensorReading
  
  ↓ build_graph_hybrid() ↓
  
  Data object (homogeneous):
    x: [N, 29]
    edge_index: [2, E]
    edge_attr: [E, 48-116]


Phase 3.3 (Hetero Incidence):
  Same HybridInferenceRequest!
  ↓ build_graph_hetero() ↓
  
  HeteroData object (bipartite):
    ["component"].x: [num_comp, 29]
    ["line"].x: [num_lines, 48-116]
    
    [("component", "source", "line")].edge_index: [2, E_src]
    [("line", "sink", "component")].edge_index: [2, E_sink]


✅ NO SENSOR DATA CHANGE REQUIRED!
✅ Same input schema (HybridInferenceRequest)
✅ Same data collection procedures
✅ Only graph representation differs
```

---

## 🚀 **Migration Timeline**

```
Week 1: GraphBuilderV2 Enhancement
  ├─ Add build_graph_hetero() method
  ├─ Create HeteroData from HybridInferenceRequest
  └─ Unit tests (>90% coverage)

Week 2: Model Architecture
  ├─ Implement HeteroComponentDiagnostics (HeteroConv-based)
  ├─ Multi-label classification head
  └─ Optional auxiliary line anomaly head

Week 3: Integration
  ├─ Update InferenceEngine.predict_hetero()
  ├─ FastAPI endpoint /v3/inference/component_diagnostics
  └─ Post-processing (logits → fault labels)

Week 4: Retraining & Validation
  ├─ Generate component-level ground truth
  ├─ Train hetero model on historical cycles
  ├─ A/B test vs Phase 3.2 edge-centric
  └─ Benchmark metrics

Week 5: Production Deployment
  ├─ Canary deployment (5% traffic)
  ├─ Monitor latency & accuracy
  └─ Full rollout
```

---

## 📚 **Related Documents**

- [COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md](./COMPONENT_DIAGNOSTICS_WITH_HETERO_GRAPH.md) — Full Phase 3.3 spec
- [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) — Phase 3.2 details (parent)
- [MODEL_CONTRACT.md](./MODEL_CONTRACT.md) — Model I/O specifications

---

**Summary:** 
Phase 3.2 gives us **excellent line sensors** (48-116D per edge). Phase 3.3 uses these sensors optimally by making **components the primary prediction targets** and organizing the graph as an **explicit bipartite incidence structure**. This improves interpretability, handles multi-port components naturally, and delivers **component-level fault predictions** (the original goal) instead of just line-level anomalies. 🎯
