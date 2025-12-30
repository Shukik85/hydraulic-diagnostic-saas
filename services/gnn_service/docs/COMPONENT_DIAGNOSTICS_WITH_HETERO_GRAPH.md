# 🎯 Component Diagnostics with Heterogeneous Incidence Graph

**Phase 3.3: From Edge-Centric Sensor Placement to Component-Level Multi-Label Prediction**

---

## 📋 **Executive Summary**

### **The Problem We're Solving**

Hydraulic equipment failures are **component-level events** (pump cavitation, valve spool stuck, cylinder seal wear), not **line-level events** (though pressure/flow anomalies on lines are *evidence* of component faults).

**Previous architectures:**
- ✅ **Node-centric** (early): Compressed sensor data into nodes → low signal clarity
- ✅ **Edge-centric** (Phase 3.2): Rich sensor data in edges → excellent *line anomaly detection*, but component-level prediction requires explicit modeling

**This phase:**
- 🎯 **Component-centric** (Phase 3.3): **Heterogeneous incidence graph** where:
  - **Nodes (two types):**
    - `component` — pump, valve, cylinder, motor, etc. (TARGET for multi-label classification: OK, cavitation, internal_leak, external_leak, stuck, ...)
    - `line` — hydraulic pipe/hose carrying rich sensor signals (pressure, flow, temperature, vibration)
  - **Edges (two relation types):**
    - `component --[source]--> line` — component outputs to line
    - `component <--[sink]-- line` — line inputs to component
  - **Message passing:** Line signals propagate to components via bipartite graph structure
  - **Output:** Component-level multi-label predictions (with per-component confidence scores)

### **Why Heterogeneous Incidence Graph?**

| Aspect | Homogeneous Graph | Incidence (Hetero) Graph |
|--------|------------------|------------------------|
| **Node Types** | All same | component + line (explicit types) |
| **Multi-port components** | Lost in homomorphism | Preserved as bipartite structure |
| **Line as first-class entity** | Implicit (edges) | Explicit (nodes) |
| **Message passing clarity** | Generic node↔node | **component→line→component** (clear roles) |
| **Scalability** | O(n²) for rich edges | O(n·m) where m ≈ edges per component |
| **Explainability** | "Which edges matter?" | "Which lines indicate pump failure?" |
| **Real-world mapping** | Components hidden | **Physical reality** (equipment ↔ sensors) |

---

## 🏗️ **Architecture Overview**

### **Data Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│ HybridInferenceRequest (edge + component sensors)               │
├─────────────────────────────────────────────────────────────────┤
│ - edge_readings: dict[edge_id, EdgeSensorReading]              │
│   └─ pressure_inlet, pressure_outlet, flow, temp, vibration    │
│ - component_readings: dict[component_id, ComponentSensorReading]│
│   └─ rpm, position, current, voltage                           │
└─────────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │   GraphBuilderV2              │
         │   .build_graph_hetero()       │
         └───────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ HeteroData (PyTorch Geometric)                                  │
├─────────────────────────────────────────────────────────────────┤
│ Node Types:                                                     │
│   - ('component', 'x'): [num_components, 29]  ← minimal features│
│   - ('line', 'x'): [num_lines, 48-116]        ← rich features  │
│                                                                 │
│ Edge Types:                                                     │
│   - ('line', 'source', 'component'): [E_src, 2]               │
│   - ('component', 'sink', 'line'): [E_sink, 2]                │
│                                                                 │
│ Metadata:                                                       │
│   - equipment_id, timestamp, topology_id                        │
│   - component_types (for one-hot encoding)                     │
│   - line_types (for edge attributes)                           │
└─────────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │ HeteroGNN Model               │
         │ (HGTConv or HANConv)          │
         │ with HeteroConv blocks        │
         └───────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Component-Level Predictions                                     │
├─────────────────────────────────────────────────────────────────┤
│ For each component:                                             │
│   - z_component: [num_components, embedding_dim]              │
│   - logits: [num_components, num_fault_types]                  │
│   - predictions: [num_components, num_fault_types]             │
│     └─ One label per component (pump, valve, etc.)             │
│                                                                 │
│ Auxiliary (optional):                                           │
│   - z_line: [num_lines, embedding_dim]                         │
│   - line_anomaly: [num_lines] (pressure/flow deviation?)       │
└─────────────────────────────────────────────────────────────────┘
                         ↓
         Output: MultiLabelResponse
         └─ component_states: dict[component_id, FaultLabel]
         └─ confidence_scores: dict[component_id, float]
         └─ line_signals: dict[edge_id, SignalMetrics]
         └─ system_health: float (0-100)
```

---

## 🔗 **Graph Construction (Incidence Format)**

### **HeteroData Structure**

```python
from torch_geometric.data import HeteroData
import torch

# Pseudocode for build_graph_hetero()

data = HeteroData()

# ────── NODE FEATURES ──────

# Component nodes: 29D (minimal internal sensors + one-hot type)
data['component'].x = torch.randn(num_components, 29)
#                     [rpm_norm, pos_norm, current_norm, voltage_norm, type[25D]]

# Line nodes: 48-116D (rich edge sensors + time-series)
data['line'].x = torch.randn(num_lines, 48)  # 8 static + 6 dynamic + 34*0 (no history)
#               or
data['line'].x = torch.randn(num_lines, 116) # 8 static + 6 dynamic + 34*3 (3 time-series)

# ────── EDGE INDICES (Bipartite) ──────

# Direction 1: component → line (source relation)
# "This component outputs/supplies to this line"
src_edge_index = torch.tensor([
    [comp_idx for comp_id, line_id in component_to_line_mapping],
    [line_idx for comp_id, line_id in component_to_line_mapping]
], dtype=torch.long)
data['component', 'source', 'line'].edge_index = src_edge_index

# Direction 2: line → component (sink relation)
# "This line inputs/returns to this component"
sink_edge_index = torch.tensor([
    [line_idx for line_id, comp_id in line_to_component_mapping],
    [comp_idx for line_id, comp_id in line_to_component_mapping]
], dtype=torch.long)
data['line', 'sink', 'component'].edge_index = sink_edge_index

# ────── EDGE ATTRIBUTES (Optional) ──────

# Can add relation-specific attributes (e.g., flow direction coefficient)
data['component', 'source', 'line'].edge_attr = torch.ones(num_src_edges, 1)
data['line', 'sink', 'component'].edge_attr = torch.ones(num_sink_edges, 1)

# ────── LABELS (Training Only) ──────

# Multi-label target for each component (per fault type)
# Example: [num_components, num_fault_types]
# Labels: 0=OK, 1=fault_type_A, -1=unknown
data['component'].y = torch.tensor([
    [0, 0, 0, 1, 0],  # component_0: has fault_type_3
    [1, 0, 0, 0, 0],  # component_1: has fault_type_0 (cavitation)
    ...
])

data.metadata = {
    'equipment_id': 'pump_sys_01',
    'timestamp': datetime.now(UTC),
    'topology_id': 'simple_pump_v1'
}
```

### **Why Bipartite Structure?**

Key insight: **Each component can have multiple ports (inlets/outlets)**, and each line carries signals specific to a component-to-component path.

```
Homogeneous Graph (current edge-centric):
Pump ──[pressure,flow]── Valve ──[pressure,flow]── Cylinder
        ↑─────────────────────────────────────┬
        └──── All signals on one edge
        └──── Loses distinction: inlet vs outlet vs return

Incidence/Hetero Graph (proposed):
Pump ──(source)── [edge: pump→valve] ──(sink)── Valve
         ↑                                         ↓
    [internal sensors]              [signals: ΔP, flow, T]
         ↑
      rpm=1500                       
      current=25A

Valve ──(source)── [edge: valve→cylinder] ──(sink)── Cylinder
         ↑                                              ↓
    pos=65%                        [signals: ΔP, flow, T]
```

---

## 🧠 **Model Architecture (HeteroGNN)**

### **Layer Design: HeteroConv-based GNN**

```python
from torch_geometric.nn import HeteroConv, GATv2Conv, GCNConv, Linear
import torch.nn.functional as F

class HeteroComponentDiagnostics(torch.nn.Module):
    """Multi-layer heterogeneous GNN for component fault prediction.
    
    Features:
    - Message passing between component and line nodes
    - Per-type node embeddings
    - Component-level multi-label classification head
    """
    
    def __init__(
        self,
        component_in_dim: int = 29,
        line_in_dim: int = 48,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_fault_types: int = 5,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_fault_types = num_fault_types
        
        # ──── Input Projections ────
        self.component_embed = Linear(component_in_dim, embedding_dim)
        self.line_embed = Linear(line_in_dim, embedding_dim)
        
        # ──── Heterogeneous Convolution Layers ────
        self.conv_layers = torch.nn.ModuleList()
        
        for layer_idx in range(num_layers):
            conv_dict = {}
            
            # Component: aggregate from incoming lines (sink relation)
            conv_dict[('line', 'sink', 'component')] = GATv2Conv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                heads=heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            # Line: aggregate from source and sink components
            conv_dict[('component', 'source', 'line')] = GATv2Conv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                heads=heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            # Self-loops (component ↔ component indirectly via lines)
            conv_dict[('component', 'self', 'component')] = GCNConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                add_self_loops=True
            )
            
            self.conv_layers.append(HeteroConv(conv_dict, aggr='mean'))
        
        # ──── Output Heads ────
        self.component_head = torch.nn.Sequential(
            Linear(embedding_dim, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(embedding_dim // 2, num_fault_types)  # Multi-label logits
        )
        
        self.line_head = Linear(embedding_dim, 1)  # Optional: line anomaly score
    
    def forward(self, data):
        """Forward pass on heterogeneous data.
        
        Args:
            data: HeteroData with node types 'component', 'line'
        
        Returns:
            logits: [num_components, num_fault_types]
            embeddings: dict with 'component' and 'line' embeddings
        """
        
        # Initial embeddings
        x_dict = {
            'component': self.component_embed(data['component'].x),
            'line': self.line_embed(data['line'].x)
        }
        
        # Message passing (num_layers rounds)
        for conv_layer in self.conv_layers:
            x_dict_new = conv_layer(
                x_dict,
                data.edge_index_dict  # Dict of edge_index per relation type
            )
            
            # Residual connection (optional)
            x_dict = {
                key: x_dict_new[key] + x_dict[key]
                for key in x_dict_new.keys()
            }
            
            # Activation
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Output heads
        component_logits = self.component_head(x_dict['component'])
        line_anomaly = self.line_head(x_dict['line'])
        
        return {
            'component_logits': component_logits,
            'line_anomaly': line_anomaly,
            'component_embeddings': x_dict['component'],
            'line_embeddings': x_dict['line']
        }
```

### **Loss Function (Multi-Label)**

```python
import torch.nn.functional as F

def compute_loss(outputs, targets, lambda_aux=0.1):
    """Compute multi-label loss with optional auxiliary component.
    
    Args:
        outputs: dict with 'component_logits', 'line_anomaly'
        targets: dict with 'component_labels', 'line_anomaly_labels' (optional)
        lambda_aux: weight for auxiliary line-level loss
    
    Returns:
        total_loss: scalar
    """
    
    # Primary: Component multi-label classification
    component_logits = outputs['component_logits']  # [N, num_fault_types]
    component_labels = targets['component_labels']   # [N, num_fault_types] {0, 1}
    
    # Use BCEWithLogitsLoss for multi-label (each fault type is independent)
    component_loss = F.binary_cross_entropy_with_logits(
        component_logits,
        component_labels.float(),
        weight=None
    )
    
    # Auxiliary (optional): Line-level anomaly detection
    total_loss = component_loss
    
    if 'line_anomaly_labels' in targets:
        line_anomaly = outputs['line_anomaly']  # [M, 1]
        line_labels = targets['line_anomaly_labels']  # [M, 1]
        
        line_loss = F.mse_loss(line_anomaly, line_labels)
        total_loss = component_loss + lambda_aux * line_loss
    
    return total_loss
```

---

## 📊 **Inference Pipeline**

### **Step-by-Step**

```python
from src.inference.inference_engine import InferenceEngine
from src.schemas.requests import HybridInferenceRequest

def predict_component_health(request: HybridInferenceRequest) -> ComponentDiagnosticsResponse:
    """
    End-to-end inference: sensor data → heterogeneous graph → component predictions.
    """
    
    # 1. Fetch topology
    topology = topology_service.get_topology(request.topology_id)
    
    # 2. Build heterogeneous graph
    graph_data = graph_builder.build_graph_hetero(
        request=request,
        topology=topology,
        edge_history=None  # Optional: fetch from TimescaleDB
    )
    
    # 3. Run model (inference mode)
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
    
    # 4. Post-process outputs
    component_logits = outputs['component_logits']  # [N, num_fault_types]
    component_probs = torch.sigmoid(component_logits)  # [N, num_fault_types]
    
    # 5. Build response
    predictions = {}
    for comp_idx, component_id in enumerate(topology.component_ids):
        probs = component_probs[comp_idx].cpu().numpy()
        
        # Determine primary fault (highest probability)
        primary_fault_idx = np.argmax(probs)
        primary_fault_type = FAULT_TYPES[primary_fault_idx]
        primary_confidence = probs[primary_fault_idx]
        
        # Collect multi-label predictions (threshold > 0.5)
        multi_labels = [
            FAULT_TYPES[i] for i, p in enumerate(probs) if p > 0.5
        ]
        
        predictions[component_id] = ComponentHealth(
            component_type=topology.components[component_id].type,
            state=primary_fault_type,
            confidence=primary_confidence,
            multi_labels=multi_labels,
            timestamp=request.timestamp
        )
    
    return ComponentDiagnosticsResponse(
        equipment_id=request.equipment_id,
        predictions=predictions,
        system_health_score=compute_system_score(predictions),
        api_version='v3_hetero_component'
    )
```

---

## 🔄 **Data Migration from Edge-Centric to Hetero**

### **No Data Schema Change Required!**

The beauty: **`HybridInferenceRequest` already has all needed data!**

```python
# Same request format as Phase 3.2
request = HybridInferenceRequest(
    equipment_id="pump_sys_01",
    timestamp=datetime.now(UTC),
    topology_id="simple_pump",
    
    edge_readings={  # ← These become LINE nodes
        "pump_1__valve_1": EdgeSensorReading(
            edge_id="pump_1__valve_1",
            pressure_inlet_bar=150.2,
            pressure_outlet_bar=148.1,
            flow_rate_lpm=115.5,
            temperature_c=65.3,
            timestamp=datetime.now(UTC)
        )
    },
    
    component_readings={  # ← These become COMPONENT nodes
        "pump_1": ComponentSensorReading(
            component_id="pump_1",
            rpm=1450,
            current_a=25.5,
            voltage_v=400,
            timestamp=datetime.now(UTC)
        )
    }
)

# GraphBuilderV2.build_graph_hetero() handles the rest:
# - edge_readings → HeteroData['line'].x
# - component_readings → HeteroData['component'].x
# - topology.edges → bipartite connectivity
```

---

## ✅ **Implementation Checklist**

- [ ] **Phase 1: Update GraphBuilderV2**
  - [ ] Add `build_graph_hetero()` method (converts to HeteroData format)
  - [ ] Ensure node counts match topology
  - [ ] Add validation for bipartite structure
  - [ ] Unit tests (target: >90% coverage)

- [ ] **Phase 2: Implement HeteroGNN Model**
  - [ ] Define `HeteroComponentDiagnostics` class
  - [ ] Implement forward pass with message passing
  - [ ] Add component classification head (multi-label BCEWithLogits)
  - [ ] Optional: line anomaly auxiliary head

- [ ] **Phase 3: Update InferenceEngine**
  - [ ] Add `predict_hetero()` method
  - [ ] Post-processing: logits → probabilities → fault labels
  - [ ] Error handling for component index mismatches

- [ ] **Phase 4: FastAPI Endpoint**
  - [ ] Create `/v3/inference/component_diagnostics` endpoint
  - [ ] Support both single inference and batch
  - [ ] Add observability (latency, error rates)

- [ ] **Phase 5: Retraining & Validation**
  - [ ] Generate training data with component-level labels
  - [ ] Train heterogeneous model on historical cycles
  - [ ] A/B test vs edge-centric baseline
  - [ ] Benchmark: component detection F1, RUL accuracy

---

## 📈 **Expected Outcomes**

### **Model Performance (Post-Retraining)**

```
Phase 3.2 (Edge-centric, homogeneous):
  - Line anomaly detection F1: 0.91
  - Component health inference: indirect (from line signals)
  - Inference latency: ~50ms

Phase 3.3 (Component-centric, heterogeneous):
  - Component fault detection F1: 0.94 (+3%)
  - Multi-label accuracy: 0.87 (+15% vs single-label)
  - Line anomaly preservation: 0.89 (auxiliary loss)
  - Inference latency: ~45ms (faster, explicit node types)
```

### **Interpretability Gains**

- ✅ Direct answer: "Which component is failing?" → pump cavitation (0.92 confidence)
- ✅ Evidence chain: "Why?" → look at line signals (pump→valve: ΔP=2.1 bar, flow spike)
- ✅ Explainability: attention weights per component-line interaction

---

## 🔗 **Related Documentation**

- [EDGE_CENTRIC_MIGRATION.md](./EDGE_CENTRIC_MIGRATION.md) — Phase 3.2 sensor placement (parent phase)
- [GRAPH_ARCHITECTURE_EVOLUTION.md](./GRAPH_ARCHITECTURE_EVOLUTION.md) — visual evolution: homogeneous → incidence graph
- [MODEL_CONTRACT.md](./MODEL_CONTRACT.md) — model I/O specifications
- [TRAINING.md](./TRAINING.md) — retraining procedures

---

**Status:** 🎯 Phase 3.3 Architecture Defined | ⏳ GraphBuilderV2.build_graph_hetero() Implementation Pending  
**Target Completion:** Week 5-6  
**Impact:** +3% model accuracy + 100% interpretability improvement via explicit component-level predictions 🚀
