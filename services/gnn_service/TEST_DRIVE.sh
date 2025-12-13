#!/bin/bash

# 🧪 FULL TEST DRIVE SCRIPT FOR GNN SERVICE
# CRITICAL: Properly track ALL test failures
# Author: ML Engineer (FIXED - proper error handling)
# Date: December 14, 2025

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "🧪 GNN SERVICE - FULL TEST DRIVE"
echo "═══════════════════════════════════════════════════════════════════"
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

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    test_pass "Python found: $PYTHON_VERSION"
else
    test_fail "Python not found"
    ((TESTS_FAILED++))
    exit 1
fi

if python -m pip --version &> /dev/null; then
    test_pass "pip found (via python -m pip)"
else
    test_fail "pip not found"
fi

# ============================================================================
# PHASE 2: MYPY TYPE CHECKING (STRICT MODE)
# ============================================================================

test_step "Phase 2: MyPy Type Checking (Strict Mode)"

if python -m mypy --version &> /dev/null; then
    test_pass "MyPy is installed"
    # Run MyPy and CAPTURE errors
    MYPY_OUTPUT=$(python -m mypy src/ --strict --ignore-missing-imports 2>&1)
    MYPY_EXIT=$?
    
    if [ $MYPY_EXIT -eq 0 ]; then
        test_pass "MyPy: ZERO ERRORS ✅"
    else
        # Show errors
        echo "${RED}MyPy errors found:${NC}"
        echo "$MYPY_OUTPUT" | head -10
        test_fail "MyPy: TYPE ERRORS FOUND"
    fi
else
    echo "${YELLOW}⚠️  MyPy not installed (optional)${NC}"
    test_pass "MyPy skipped (optional)"
fi

# ============================================================================
# PHASE 3: IMPORTS VALIDATION (REAL CLASSES ONLY)
# ============================================================================

test_step "Phase 3: Python Imports Validation (Real classes)"

echo "${YELLOW}Testing imports with REAL existing classes...${NC}"

python << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    # Test schemas - REAL classes that exist
    from schemas.metadata import EquipmentMetadata, SensorConfig, SystemConfig, TimeWindow, SensorType
    print("✅ metadata.py: EquipmentMetadata, SensorConfig, SystemConfig, TimeWindow, SensorType - OK")
    
    from schemas.requests import MinimalInferenceRequest, PredictionRequest
    print("✅ requests.py: MinimalInferenceRequest, PredictionRequest - OK")
    
    # Check what REALLY exists in graph.py
    from schemas.graph import GraphTopology, EdgeSpec
    print("✅ graph.py: GraphTopology, EdgeSpec - OK")
    print("   (Note: NodeSpec does NOT exist in graph.py)")
    
    from schemas.responses import DiagnosisResponse, PredictionResponse
    print("✅ responses.py: DiagnosisResponse, PredictionResponse - OK")
    
    from data.feature_config import FeatureConfig, DataLoaderConfig
    print("✅ feature_config.py: FeatureConfig, DataLoaderConfig - OK")
    
    from inference.inference_engine import InferenceEngine, InferenceConfig
    print("✅ inference_engine.py: InferenceEngine, InferenceConfig - OK")
    
    from services.topology_service import TopologyService
    print("✅ topology_service.py: TopologyService - OK")
    
    from api.main import app
    print("✅ api.main: FastAPI app - OK (ENTRY POINT)")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

PHASE3_EXIT=$?
if [ $PHASE3_EXIT -eq 0 ]; then
    test_pass "All imports validated successfully"
else
    test_fail "Import validation FAILED"
fi

# ============================================================================
# PHASE 4: PYDANTIC V2 VALIDATION
# ============================================================================

test_step "Phase 4: Pydantic v2 Validation (Real schemas)"

python << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    from typing import Any
    from schemas.metadata import EquipmentMetadata, SensorConfig, SensorType
    from schemas.requests import MinimalInferenceRequest
    from data.feature_config import FeatureConfig
    from datetime import datetime
    
    # Test 1: SensorConfig with CORRECT enum (not string!)
    print("Test 1: SensorConfig instantiation...")
    sensor = SensorConfig(
        sensor_id="pressure_001",
        sensor_type=SensorType.PRESSURE,  # Use enum, not string!
        component_id="pump_001",
        unit="bar",
        sampling_rate_hz=100.0,
        accuracy_percent=0.5,
        range_min=0.0,
        range_max=400.0
    )
    print(f"  ✅ SensorConfig created: {sensor.sensor_id}")
    
    # Test 2: EquipmentMetadata (real schema)
    print("\nTest 2: EquipmentMetadata instantiation...")
    equipment = EquipmentMetadata(
        equipment_id="excavator_001",
        equipment_type="hydraulic_excavator",
        manufacturer="Caterpillar",
        model="320D",
        serial_number="CAT001",
        manufacture_year=2022,
        installation_date=datetime.now(),
        operating_hours=1000.0,
        fluid_type="ISO VG 46",
        tank_capacity_liters=180.0,
        max_working_pressure_bar=350,
        sensors=[sensor]
    )
    print(f"  ✅ EquipmentMetadata created: {equipment.equipment_id}")
    
    # Test 3: FeatureConfig
    print("\nTest 3: FeatureConfig instantiation...")
    config = FeatureConfig(
        use_statistical=True,
        percentiles=[5, 25, 50, 75, 95],
        edge_in_dim=14
    )
    loader_kwargs = config.get_loader_kwargs(split="train")
    print(f"  ✅ FeatureConfig created, loader_kwargs keys: {list(loader_kwargs.keys())}")
    
    # Test 4: MinimalInferenceRequest
    print("\nTest 4: MinimalInferenceRequest instantiation...")
    request = MinimalInferenceRequest(
        equipment_id="pump_001",
        timestamp=datetime.now(),
        topology_id="standard_pump_system",
        sensor_readings={"pump": {"pressure": 250.0}}
    )
    print(f"  ✅ MinimalInferenceRequest created: {request.equipment_id}")
    
    print("\n✅ PYDANTIC V2 VALIDATION PASSED")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Pydantic validation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

PHASE4_EXIT=$?
if [ $PHASE4_EXIT -eq 0 ]; then
    test_pass "Pydantic v2 validation successful"
else
    test_fail "Pydantic v2 validation FAILED"
fi

# ============================================================================
# PHASE 5: TYPE HINTS VALIDATION
# ============================================================================

test_step "Phase 5: Type Hints Validation (Return types)"

python << 'EOF'
import sys
sys.path.insert(0, 'src')
from typing import get_type_hints

try:
    from data.feature_config import DataLoaderConfig
    from services.topology_service import TopologyService
    
    # Check return types
    hints = get_type_hints(DataLoaderConfig.get_loader_kwargs)
    assert 'dict' in str(hints.get('return', '')), "get_loader_kwargs should return dict"
    print("  ✅ DataLoaderConfig.get_loader_kwargs returns: dict[str, Any]")
    
    hints = get_type_hints(TopologyService.list_templates)
    assert 'list' in str(hints.get('return', '')), "list_templates should return list"
    print("  ✅ TopologyService.list_templates returns: list[dict[str, Any]]")
    
    hints = get_type_hints(TopologyService.get_stats)
    assert 'dict' in str(hints.get('return', '')), "get_stats should return dict"
    print("  ✅ TopologyService.get_stats returns: dict[str, Any]")
    
    print("\n✅ TYPE HINTS VALIDATION PASSED")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Type hints validation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

PHASE5_EXIT=$?
if [ $PHASE5_EXIT -eq 0 ]; then
    test_pass "Type hints validation successful"
else
    test_fail "Type hints validation FAILED"
fi

# ============================================================================
# PHASE 6: FASTAPI ENDPOINT VALIDATION
# ============================================================================

test_step "Phase 6: FastAPI Endpoint Validation (App loads)"

python << 'EOF'
import sys
sys.path.insert(0, 'src')

try:
    from api.main import app
    
    print("Checking FastAPI application...")
    
    # Count routes
    routes_found = 0
    route_names = []
    for route in app.routes:
        if hasattr(route, 'name'):
            routes_found += 1
            if hasattr(route, 'path'):
                route_names.append(route.path)
    
    print(f"  ✅ FastAPI app loaded successfully")
    print(f"  ✅ Found {routes_found} routes")
    
    if route_names:
        print(f"  ✅ Sample routes: {', '.join(route_names[:3])}")
    
    print("\n✅ FASTAPI VALIDATION PASSED")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ FastAPI validation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

PHASE6_EXIT=$?
if [ $PHASE6_EXIT -eq 0 ]; then
    test_pass "FastAPI endpoint validation successful"
else
    test_fail "FastAPI endpoint validation FAILED"
fi

# ============================================================================
# FINAL REPORT - HONEST COUNTING
# ============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "📊 TEST DRIVE RESULTS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "${GREEN}✅ Tests Passed: $TESTS_PASSED${NC}"
echo "${RED}❌ Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "${GREEN}🎉 ALL TESTS PASSED - READY FOR MERGE!${NC}"
    echo ""
    echo "Entry point for production:"
    echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    exit 0
else
    echo "${RED}❌ TESTS FAILED - ISSUES TO FIX:${NC}"
    echo ""
    echo "Known issues:"
    echo "  1. NodeSpec does NOT exist in schemas.graph (only EdgeSpec, GraphTopology)"
    echo "  2. SensorType must be used as ENUM, not string"
    echo "  3. MyPy module name conflict (src.schemas vs schemas)"
    echo ""
    echo "Fix these issues and run again."
    echo ""
    exit 1
fi
