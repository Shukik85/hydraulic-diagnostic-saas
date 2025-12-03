# Environment Setup for Quick Validation

**Date:** 04.12.2025  
**Purpose:** Setup guide for Phase 3 quick validation  
**Time:** 5-10 minutes

---

## 🛠️ Prerequisites

- Python 3.14+ (you have this ✅)
- Virtual environment activated (you have this ✅)
- Git branch: `feature/gnn-service-production-ready`

---

## 📚 Step 1: Install Dependencies (5 min)

### **Windows (your setup):**

```bash
# Navigate to gnn_service
cd services/gnn_service

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - pandas (for CSV inspection)
# - torch, torch-geometric (for graphs)
# - All other requirements
```

**Expected output:**
```
Collecting pandas==2.2.2
...
Successfully installed pandas-2.2.2 numpy-1.26.4 ...
```

### **If PyTorch installation fails:**

```bash
# Install PyTorch 2.8 separately (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu128

# Then install PyG
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Finally, rest of requirements
pip install -r requirements.txt
```

---

## 🔧 Step 2: Set PYTHONPATH

### **Windows (Git Bash):**

```bash
# Set PYTHONPATH to current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify
echo $PYTHONPATH
# Should show: /h/hydraulic-diagnostic-saas/services/gnn_service
```

### **Windows (Command Prompt):**

```cmd
set PYTHONPATH=%PYTHONPATH%;%CD%
echo %PYTHONPATH%
```

### **Windows (PowerShell):**

```powershell
$env:PYTHONPATH += ";$(Get-Location)"
$env:PYTHONPATH
```

### **Linux/macOS:**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## ✅ Step 3: Verify Setup

```bash
# Test imports
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "from src.data.edge_features import EdgeFeatureComputer; print('EdgeFeatureComputer: OK')"
```

**Expected output:**
```
pandas: 2.2.2
torch: 2.8.0
PyG: 2.6.1
EdgeFeatureComputer: OK
```

---

## 🚀 Step 4: Run Quick Validation

### **All steps in sequence:**

```bash
# Step 1: Inspect CSV sensors
echo "=== STEP 1: Inspect CSV ==="
python scripts/inspect_csv_sensors.py \
    --input data/bim_comprehensive.csv \
    --output data/analysis

echo ""
echo "=== STEP 2: Convert Graphs ==="
python scripts/convert_graphs_to_14d.py \
    --input data/gnn_graphs_multilabel.pt \
    --edge-specs data/edge_specifications.json \
    --output data/gnn_graphs_v2_14d_test.pt \
    --max-samples 200

echo ""
echo "=== STEP 3: Test Model ==="
python test_14d_model.py

echo ""
echo "✅ VALIDATION COMPLETE!"
```

---

## 🐛 Troubleshooting

### **Issue 1: ModuleNotFoundError: No module named 'pandas'**

**Solution:**
```bash
pip install pandas==2.2.2
```

---

### **Issue 2: ModuleNotFoundError: No module named 'src'**

**Solution:**
```bash
# Make sure you're in services/gnn_service directory
pwd
# Should show: /h/hydraulic-diagnostic-saas/services/gnn_service

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify
python -c "from src.data.edge_features import EdgeFeatureComputer; print('OK')"
```

---

### **Issue 3: FileNotFoundError: data/bim_comprehensive.csv**

**Solution:**
```bash
# Check if file exists
ls -lh data/bim_comprehensive.csv

# If missing, pull from Git LFS
git lfs pull
```

---

### **Issue 4: CUDA out of memory**

**Solution:**
```bash
# Use CPU instead (slower but works)
export CUDA_VISIBLE_DEVICES=""

# Or reduce batch size in conversion script
python scripts/convert_graphs_to_14d.py ... --max-samples 50
```

---

### **Issue 5: Torch version mismatch**

**Solution:**
```bash
# Check current version
python -c "import torch; print(torch.__version__)"

# If wrong, reinstall
pip uninstall torch torchvision torchaudio torch-geometric
pip install torch==2.8.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric==2.6.1
```

---

## 📊 Expected Outputs

### **Step 1: CSV Inspection**

```
Reading CSV: data/bim_comprehensive.csv
File size: 68.6 MB

Dataset shape (first 1000 rows): (1000, 50)
Columns: 50

============================================================
SENSOR INVENTORY
============================================================

PRESSURE (9 columns):
  - pump_main_1_pressure_bar              [50.00 to 330.00] (missing: 0)
  - pump_main_2_pressure_bar              [50.00 to 330.00] (missing: 0)
  ...

TEMPERATURE (9 columns):
  - pump_main_1_temperature_c             [55.00 to 75.00] (missing: 0)
  ...

============================================================
PHASE 3 COMPATIBILITY
============================================================
  pressure_drop_bar         → ✅ AVAILABLE
  flow_rate_lpm             → ⚠️ COMPUTABLE (from other sensors)
  temperature_delta_c       → ✅ AVAILABLE
  vibration_level_g         → ❌ MISSING (use synthetic)
  age_hours                 → ⚠️ FROM METADATA
  maintenance_score         → ⚠️ FROM METADATA

✅ Saved sensor inventory: data/analysis/sensor_inventory.json
✅ Saved compatibility report: data/analysis/phase3_compatibility.json

✅ READY FOR QUICK VALIDATION
```

---

### **Step 2: Graph Conversion**

```
Loading edge specifications: data/edge_specifications.json
Loaded 8 edge specifications

Loading graphs: data/gnn_graphs_multilabel.pt
Loaded 1000+ graphs
Using first 200 samples

Converting graphs to 14D...
100%|████████████| 200/200 [00:30<00:00, 6.5it/s]

============================================================
VALIDATION
============================================================

Graph 0:
  Nodes: torch.Size([9, 34])
  Edges: torch.Size([2, 8])
  Edge features: torch.Size([8, 14])
  ✅ Valid (14D, no NaN)

Graph 1:
  Nodes: torch.Size([9, 34])
  Edges: torch.Size([2, 8])
  Edge features: torch.Size([8, 14])
  ✅ Valid (14D, no NaN)

...

Saving converted graphs: data/gnn_graphs_v2_14d_test.pt

✅ Saved 200 graphs with 14D edge features

============================================================
STATISTICS
============================================================

Dataset:
  Total graphs: 200
  Total edges: 1600
  Avg nodes/graph: 9.0
  Avg edges/graph: 8.0
  Edge feature dim: 14

✅ CONVERSION COMPLETE
```

---

### **Step 3: Model Test**

```
============================================================
PHASE 3 MODEL VALIDATION - 14D Edge Features
============================================================

Loading graphs: data/gnn_graphs_v2_14d_test.pt
Loaded 200 graphs

Graph structure:
  Nodes: torch.Size([9, 34])
  Edges: torch.Size([2, 8])
  Edge features: torch.Size([8, 14])
  ✅ Edge features are 14D
  ✅ No NaN/Inf values

Creating model...
  Model created: UniversalTemporalGNN
  Edge dim: 14

Running forward pass...
  ✅ Forward pass successful!

Validating outputs...
  Health: torch.Size([1, 1]) = 0.823
  Degradation: torch.Size([1, 1]) = 0.156
  Anomaly: torch.Size([1, 9])
  ✅ Output shapes correct
  ✅ Health in valid range: 0.823
  ✅ Degradation in valid range: 0.156
  ✅ No NaN in outputs

Testing batch inference (5 graphs)...
  Batch health: torch.Size([5, 1])
  Batch degradation: torch.Size([5, 1])
  Batch anomaly: torch.Size([5, 9])
  ✅ Batch inference successful!

============================================================
VALIDATION SUMMARY
============================================================

✅ ALL TESTS PASSED!

Phase 3.1 Components Validated:
  ✅ 14D edge features loaded
  ✅ Model accepts 14D edges
  ✅ Forward pass successful
  ✅ Output shapes correct
  ✅ Output values in valid range
  ✅ Batch inference works

🚀 Ready for:
  - Full dataset conversion
  - Model retraining (v2.0.0)
  - Production deployment
```

---

## 🎉 Success!

If you see all ✅ checkmarks, validation is complete!

**Next steps:**
1. Full dataset conversion (1000+ samples)
2. Model retraining (6-8 hours)
3. Production deployment

---

## 📞 Support

- GitHub Issues: #118
- README.md
- Quick Validation Guide (workspace/quick_validation_guide.md)

---

**Last Updated:** 04.12.2025, 00:30 MSK  
**Version:** 1.0.0  
**Status:** Ready for execution
