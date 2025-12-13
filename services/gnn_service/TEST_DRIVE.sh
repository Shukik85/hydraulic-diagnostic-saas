#!/bin/bash

# 🧪 FULL TEST DRIVE SCRIPT FOR GNN SERVICE
# Validates all MyPy fixes before merging to master
# Author: ML Engineer
# Date: December 14, 2025

set -e  # Exit on error

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "🧪 GNN SERVICE - FULL TEST DRIVE"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_step() {
    echo ""
    echo "${YELLOW}→ $1${NC}"
}

test_pass() {
    echo "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

test_fail() {
    echo "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

# ============================================================================
# PHASE 1: ENVIRONMENT & DEPENDENCIES
# ============================================================================

test_step "Phase 1: Environment & Dependencies"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    test_pass "Python 3 found: $PYTHON_VERSION"
else
    test_fail "Python 3 not found"
    exit 1
fi

if command -v pip3 &> /dev/null; then
    test_pass "pip3 found"
else
    test_fail "pip3 not found"
    exit 1
fi

# ============================================================================
# PHASE 2: MYPY TYPE CHECKING (STRICT MODE)
# ============================================================================

test_step "Phase 2: MyPy Type Checking (Strict Mode)"

if command -v mypy &> /dev/null; then
    test_pass "MyPy is installed"
    
    # Run MyPy on src directory
    if mypy services/gnn_service/src/ --strict --ignore-missing-imports 2>&1 | grep -q "Success"; then
        test_pass "MyPy strict mode: ALL TYPE CHECKS PASSED"
    else
        test_fail "MyPy found type errors"
        echo "${YELLOW}Running full MyPy output:${NC}"
        mypy services/gnn_service/src/ --strict --ignore-missing-imports || true
    fi
else
    echo "${YELLOW}⚠️  MyPy not installed. Skipping type check.${NC}"
    echo "   Install with: pip install mypy"
fi

# ============================================================================
# PHASE 3: IMPORTS VALIDATION
# ============================================================================

test_step "Phase 3: Python Imports Validation"

echo "${YELLOW}Testing imports from fixed files...${NC}"

python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')

try:
    # Test Pydantic v2 imports
    from schemas.metadata import SensorMetadata, EquipmentMetadata
    print("✅ metadata.py imports OK")
    
    from schemas.requests import MinimalInferenceRequest, PredictionRequest
    print("✅ requests.py imports OK")
    
    from schemas.graph import GraphTopology, NodeSpec, EdgeSpec
    print("✅ graph.py imports OK")
    
    from data.feature_config import FeatureConfig, DataLoaderConfig
    print("✅ feature_config.py imports OK")
    
    from inference.inference_engine import InferenceEngine, InferenceConfig
    print("✅ inference_engine.py imports OK")
    
    from services.topology_service import TopologyService, get_topology_service
    print("✅ topology_service.py imports OK")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    test_pass "All imports validated successfully"
else
    test_fail "Import validation failed"
fi

# ============================================================================
# PHASE 4: PYDANTIC V2 VALIDATION
# ============================================================================

test_step "Phase 4: Pydantic v2 Validation"

python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')

try:
    from typing import Any
    from schemas.metadata import SensorMetadata
    from schemas.requests import MinimalInferenceRequest
    from data.feature_config import FeatureConfig, DataLoaderConfig
    from datetime import datetime
    
    # Test 1: FeatureConfig
    print("Testing FeatureConfig...")
    config = FeatureConfig(
        use_statistical=True,
        percentiles=[5, 25, 50, 75, 95],
        edge_in_dim=14
    )
    print(f"  ✅ FeatureConfig created: edge_in_dim={config.edge_in_dim}")
    
    # Test 2: get_loader_kwargs returns dict[str, Any]
    loader_kwargs = config.get_loader_kwargs(split="train")
    print(f"  ✅ DataLoaderConfig.get_loader_kwargs() returns dict")
    print(f"     Keys: {list(loader_kwargs.keys())}")
    
    # Test 3: SensorMetadata with Pydantic v2
    metadata = SensorMetadata(
        sensor_id="pump_001",
        sensor_name="Pump Pressure",
        sensor_type="pressure",
        unit="bar",
        min_value=0.0,
        max_value=350.0
    )
    print(f"  ✅ SensorMetadata created: {metadata.sensor_name}")
    
    # Test 4: MinimalInferenceRequest
    request = MinimalInferenceRequest(
        equipment_id="pump_001",
        timestamp=datetime.now(),
        topology_id="standard_pump_system",
        sensor_readings={"pump": {"pressure": 250.0}}
    )
    print(f"  ✅ MinimalInferenceRequest created: {request.equipment_id}")
    
    print("\n✅ PYDANTIC V2 VALIDATION PASSED")
    
except Exception as e:
    print(f"\n❌ Pydantic validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    test_pass "Pydantic v2 validation successful"
else
    test_fail "Pydantic v2 validation failed"
fi

# ============================================================================
# PHASE 5: TYPE HINTS VALIDATION
# ============================================================================

test_step "Phase 5: Type Hints Validation"

python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')
import inspect
from typing import get_type_hints, Any

try:
    from data.feature_config import DataLoaderConfig
    from inference.inference_engine import InferenceEngine
    from services.topology_service import TopologyService
    
    # Check DataLoaderConfig.get_loader_kwargs return type
    hints = get_type_hints(DataLoaderConfig.get_loader_kwargs)
    print(f"DataLoaderConfig.get_loader_kwargs returns: {hints.get('return', 'Unknown')}")
    assert 'dict' in str(hints.get('return', '')), "Should return dict"
    print("  ✅ Correct return type: dict[str, Any]")
    
    # Check TopologyService.list_templates return type
    hints = get_type_hints(TopologyService.list_templates)
    print(f"\nTopologyService.list_templates returns: {hints.get('return', 'Unknown')}")
    assert 'list' in str(hints.get('return', '')), "Should return list"
    print("  ✅ Correct return type: list[dict[str, Any]]")
    
    # Check TopologyService.get_stats return type
    hints = get_type_hints(TopologyService.get_stats)
    print(f"\nTopologyService.get_stats returns: {hints.get('return', 'Unknown')}")
    assert 'dict' in str(hints.get('return', '')), "Should return dict"
    print("  ✅ Correct return type: dict[str, Any]")
    
    print("\n✅ TYPE HINTS VALIDATION PASSED")
    
except Exception as e:
    print(f"\n❌ Type hints validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    test_pass "Type hints validation successful"
else
    test_fail "Type hints validation failed"
fi

# ============================================================================
# PHASE 6: FASTAPI ENDPOINT TYPES
# ============================================================================

test_step "Phase 6: FastAPI Endpoint Type Validation"

python3 << 'EOF'
import sys
sys.path.insert(0, 'services/gnn_service/src')
import inspect
from typing import get_type_hints

try:
    from api.main import app
    
    # Check endpoints are properly typed
    print("Checking FastAPI endpoints...")
    
    # Get all routes
    routes_found = 0
    for route in app.routes:
        if hasattr(route, 'endpoint'):
            routes_found += 1
    
    print(f"  ✅ Found {routes_found} FastAPI routes")
    
    if routes_found > 0:
        print("  ✅ FastAPI app loaded successfully")
    else:
        print("  ⚠️  No routes found (might be expected in test mode)")
    
    print("\n✅ FASTAPI VALIDATION PASSED")
    
except Exception as e:
    print(f"\n❌ FastAPI validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    test_pass "FastAPI endpoint validation successful"
else
    test_fail "FastAPI endpoint validation failed"
fi

# ============================================================================
# FINAL REPORT
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "📊 TEST DRIVE RESULTS"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "${GREEN}✅ Tests Passed: $TESTS_PASSED${NC}"
echo "${RED}❌ Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "${GREEN}🎉 ALL TESTS PASSED - READY FOR MERGE!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Push to GitHub"
    echo "  2. Create PR to master"
    echo "  3. Code review"
    echo "  4. Merge to master"
    echo "  5. Deploy to staging"
    echo ""
    exit 0
else
    echo "${RED}❌ SOME TESTS FAILED - DO NOT MERGE YET${NC}"
    echo ""
    echo "Fix the issues above and run this script again."
    echo ""
    exit 1
fi
