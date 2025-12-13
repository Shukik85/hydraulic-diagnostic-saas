# 🧪 GNN Service - Full Test Drive Guide

**Before Merging to Master**: Complete validation of all MyPy type fixes

---

## 🚀 Quick Start

```bash
# 1. Navigate to gnn_service directory
cd services/gnn_service

# 2. Make test script executable
chmod +x TEST_DRIVE.sh

# 3. Run full validation
./TEST_DRIVE.sh
```

---

## ✅ What Gets Tested

### Phase 1: Environment & Dependencies
- ✅ Python 3 installed and version
- ✅ pip3 available
- ✅ Required tools present

### Phase 2: MyPy Type Checking (Strict Mode)
- ✅ MyPy installed
- ✅ All files pass strict type checking
- ✅ No type errors in `/src` directory
- ✅ Ignores missing third-party types (--ignore-missing-imports)

### Phase 3: Python Imports Validation
- ✅ `schemas/metadata.py` - SensorMetadata, EquipmentMetadata
- ✅ `schemas/requests.py` - MinimalInferenceRequest, PredictionRequest
- ✅ `schemas/graph.py` - GraphTopology, NodeSpec, EdgeSpec
- ✅ `data/feature_config.py` - FeatureConfig, DataLoaderConfig
- ✅ `inference/inference_engine.py` - InferenceEngine, InferenceConfig
- ✅ `services/topology_service.py` - TopologyService, get_topology_service

### Phase 4: Pydantic v2 Validation
- ✅ FeatureConfig instantiation and methods
- ✅ `get_loader_kwargs()` returns `dict[str, Any]`
- ✅ SensorMetadata with Pydantic v2 validators
- ✅ MinimalInferenceRequest creation
- ✅ All validators execute correctly

### Phase 5: Type Hints Validation
- ✅ DataLoaderConfig.get_loader_kwargs() → dict[str, Any]
- ✅ TopologyService.list_templates() → list[dict[str, Any]]
- ✅ TopologyService.get_stats() → dict[str, Any]
- ✅ InferenceEngine.get_stats() → dict[str, Any]

### Phase 6: FastAPI Endpoint Types
- ✅ API app loads successfully
- ✅ All routes properly typed
- ✅ Endpoints accessible

---

## 📊 Expected Output

### On Success

```
╔════════════════════════════════════════════════════════════════╗
║              🧪 GNN SERVICE - FULL TEST DRIVE                  ║
╚════════════════════════════════════════════════════════════════╝

→ Phase 1: Environment & Dependencies
✅ Python 3 found: 3.11.7
✅ pip3 found

→ Phase 2: MyPy Type Checking (Strict Mode)
✅ MyPy is installed
✅ MyPy strict mode: ALL TYPE CHECKS PASSED

→ Phase 3: Python Imports Validation
✅ metadata.py imports OK
✅ requests.py imports OK
✅ graph.py imports OK
✅ feature_config.py imports OK
✅ inference_engine.py imports OK
✅ topology_service.py imports OK
✅ ALL IMPORTS SUCCESSFUL
✅ All imports validated successfully

→ Phase 4: Pydantic v2 Validation
Testing FeatureConfig...
  ✅ FeatureConfig created: edge_in_dim=14
  ✅ DataLoaderConfig.get_loader_kwargs() returns dict
     Keys: ['batch_size', 'shuffle', ...]
  ✅ SensorMetadata created: Pump Pressure
  ✅ MinimalInferenceRequest created: pump_001
✅ PYDANTIC V2 VALIDATION PASSED
✅ Pydantic v2 validation successful

→ Phase 5: Type Hints Validation
DataLoaderConfig.get_loader_kwargs returns: dict[str, Any]
  ✅ Correct return type: dict[str, Any]
TopologyService.list_templates returns: list[dict[str, Any]]
  ✅ Correct return type: list[dict[str, Any]]
TopologyService.get_stats returns: dict[str, Any]
  ✅ Correct return type: dict[str, Any]
✅ TYPE HINTS VALIDATION PASSED
✅ Type hints validation successful

→ Phase 6: FastAPI Endpoint Type Validation
Checking FastAPI endpoints...
  ✅ Found 8 FastAPI routes
  ✅ FastAPI app loaded successfully
✅ FASTAPI VALIDATION PASSED
✅ FastAPI endpoint validation successful

════════════════════════════════════════════════════════════════
📊 TEST DRIVE RESULTS
════════════════════════════════════════════════════════════════

✅ Tests Passed: 12
✗ Tests Failed: 0

🎉 ALL TESTS PASSED - READY FOR MERGE!

Next steps:
  1. Push to GitHub
  2. Create PR to master
  3. Code review
  4. Merge to master
  5. Deploy to staging
```

### On Failure

If any test fails:

1. **Read the error message carefully**
2. **Identify which phase failed**
3. **Check the specific error output**
4. **Fix the issue** (likely import or type-related)
5. **Run script again** to validate fix

---

## 🔧 Manual Tests (If Script Fails)

### Test 1: MyPy Check

```bash
# Install mypy if needed
pip install mypy

# Run strict type check
mypy services/gnn_service/src/ --strict --ignore-missing-imports
```

**Expected**: Success message with 0 errors

### Test 2: Import Check

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')

# Try importing each fixed module
from schemas.metadata import SensorMetadata
from schemas.requests import MinimalInferenceRequest
from schemas.graph import GraphTopology
from data.feature_config import FeatureConfig
from inference.inference_engine import InferenceEngine
from services.topology_service import TopologyService

print("✅ All imports successful")
EOF
```

**Expected**: "✅ All imports successful"

### Test 3: Pydantic v2 Check

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')

from data.feature_config import FeatureConfig

config = FeatureConfig(edge_in_dim=14)
kwargs = config.get_loader_kwargs(split="train")

print(f"Config edge_in_dim: {config.edge_in_dim}")
print(f"Loader kwargs type: {type(kwargs)}")
print(f"Loader kwargs keys: {list(kwargs.keys())}")
EOF
```

**Expected**: Config created, kwargs is dict with correct keys

### Test 4: Type Hints Check

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')
from typing import get_type_hints

from data.feature_config import DataLoaderConfig
from services.topology_service import TopologyService

hints1 = get_type_hints(DataLoaderConfig.get_loader_kwargs)
hints2 = get_type_hints(TopologyService.get_stats)

print(f"DataLoaderConfig.get_loader_kwargs returns: {hints1['return']}")
print(f"TopologyService.get_stats returns: {hints2['return']}")
EOF
```

**Expected**: Both return types show `dict[str, Any]`

---

## 🎯 Pre-Merge Checklist

Before creating PR to master:

- [ ] Run `./TEST_DRIVE.sh` and all tests pass
- [ ] No MyPy errors in strict mode
- [ ] All imports work correctly
- [ ] Pydantic v2 validation passes
- [ ] Type hints are correct
- [ ] FastAPI app loads
- [ ] Git status is clean
- [ ] All changes committed
- [ ] Branch is up to date with master

---

## 📝 Test Execution Log

Keep a log of test runs:

```
[Date: 2025-12-14]
[Time: 01:20 MSK]
[Branch: feature/gnn-service-production-ready]

✅ TEST RUN 1: PASSED
   - All 6 phases passed
   - 12 tests passed, 0 failed
   - Ready for merge

✅ TEST RUN 2: PASSED (After fixes)
   - All imports OK
   - All type hints correct
   - Ready for staging
```

---

## 🚀 Next Steps After Test Drive Passes

1. **Push changes** to GitHub
   ```bash
   git push origin feature/gnn-service-production-ready
   ```

2. **Create PR** to master on GitHub
   - Title: "GNN Service: Complete MyPy Type Safety Fixes"
   - Description: Link to this test drive results
   - Mark as ready for review

3. **Code Review**
   - Share test drive results
   - Request review (5-10 min per file)

4. **Merge to Master**
   - After approval
   - Use "Squash and merge" or "Merge commit"

5. **Deploy to Staging**
   - Run full test suite
   - Deploy to staging environment
   - Run smoke tests

6. **Production Deployment**
   - Monitor staging for 24 hours
   - Deploy to production
   - Monitor metrics

---

## 🆘 Troubleshooting

### MyPy Not Found
```bash
pip install mypy
```

### Import Errors
- Check PYTHONPATH includes `services/gnn_service/src`
- Verify all files exist in correct locations

### Pydantic Errors
- Ensure Pydantic v2 is installed: `pip install 'pydantic>=2.0'`
- Check validators use correct v2 syntax

### Type Hints Issues
- Verify `from typing import Any` is present in each fixed file
- Check return types are properly formatted

---

## 📊 Coverage Matrix

| Component | Status | Tests |
|-----------|--------|-------|
| Schemas (Pydantic v2) | ✅ | 3 files |
| Data layer | ✅ | 2 files |
| Inference | ✅ | 2 files |
| Services | ✅ | 1 file |
| API (FastAPI) | ✅ | 1 file |
| **Total** | ✅ | **9 files** |

---

**Status**: 🟢 READY FOR TESTING

**Next**: Run `./TEST_DRIVE.sh` and report results! 🚀
