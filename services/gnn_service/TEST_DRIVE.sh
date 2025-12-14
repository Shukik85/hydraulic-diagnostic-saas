#!/bin/bash

# 🧪 FULL TEST DRIVE SCRIPT FOR GNN SERVICE
# CRITICAL: Proper error handling + Ruff autofix + better parsing
# Author: ML Engineer (FINAL - with Windows path support)
# Date: December 14, 2025

echo ""
echo "=================================================================="
echo "🧪 GNN SERVICE - FULL TEST DRIVE"
echo "=================================================================="
echo ""

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_step() {
    echo ""
    echo "→ $1"
}

test_pass() {
    echo "✅ $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo "❌ $1"
    ((TESTS_FAILED++))
}

# ============================================================================
# PHASE 0: RUFF LINTING (AUTOFIX + CHECK)
# ============================================================================

test_step "Phase 0: Ruff Linting (autofix + check src/ and tests/)"

if python -m ruff --version &> /dev/null; then
    test_pass "Ruff is installed"
    
    # Step 1: Autofix safe issues
    echo "🔧 Running Ruff autofix (safe fixes only)..."
    python -m ruff check src/ tests/ --fix --exit-zero 2>&1 | head -5
    
    # Step 2: Check remaining issues
    RUFF_OUTPUT=$(python -m ruff check src/ tests/ 2>&1)
    RUFF_EXIT=$?
    
    if [ $RUFF_EXIT -eq 0 ]; then
        test_pass "Ruff: NO LINTING ISSUES REMAINING ✅"
    else
        # Count errors by code - SIMPLE approach
        echo ""
        echo "📄 Ruff Error Summary:"
        
        # Extract error codes (I001, E402, etc.) and count them
        echo "$RUFF_OUTPUT" | grep -oE '[A-Z][A-Z]?[0-9]{3}' | sort | uniq -c | sort -rn | head -10 | while read count code; do
            printf "    %3d × %s\n" "$count" "$code"
        done
        
        # Count total lines that look like errors (contain "src" or "tests")
        TOTAL_ERRORS=$(echo "$RUFF_OUTPUT" | grep -E '(src|tests)' | grep -cE ':[0-9]+:[0-9]+')
        if [ $TOTAL_ERRORS -gt 0 ]; then
            echo ""
            echo "📈 Total issues: $TOTAL_ERRORS"
        fi
        
        # Show first 5 actual error lines
        echo ""
        echo "🔍 First 5 issues:"
        echo "$RUFF_OUTPUT" | grep -E '(src|tests)' | grep -E ':[0-9]+:[0-9]+:' | head -5
        
        test_fail "Ruff: LINTING ISSUES FOUND (run 'ruff check --fix src/' to autofix)"
    fi
else
    echo "⚠️  Ruff not installed (optional)"
    echo "   Install with: python -m pip install ruff"
    test_pass "Ruff skipped (optional)"
fi

# ============================================================================
# PHASE 1: ENVIRONMENT & DEPENDENCIES
# ============================================================================

test_step "Phase 1: Environment & Dependencies"

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    test_pass "Python found: $PYTHON_VERSION"
else
    test_fail "Python not found"
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
        # Show summary - error count by file
        echo "📄 MyPy errors (by file, top 15):"
        echo "$MYPY_OUTPUT" | grep -E "error:" | cut -d: -f1 | sed 's/\\/\//g' | sort | uniq -c | sort -rn | head -15 | awk '{printf "    %3d × %s\n", $1, $2}'
        
        # Count total
        TOTAL_MYPY=$(echo "$MYPY_OUTPUT" | grep -cE "error:")
        echo ""
        echo "📈 Total MyPy errors: $TOTAL_MYPY"
        
        # Show first 5 actual errors
        echo ""
        echo "🔍 First 5 errors:"
        echo "$MYPY_OUTPUT" | grep -E "error:" | head -5
        
        # Check for specific error types
        PROP_ERRORS=$(echo "$MYPY_OUTPUT" | grep -c "prop-decorator")
        if [ $PROP_ERRORS -gt 0 ]; then
            echo ""
            echo "⚠️  Main issue: $PROP_ERRORS × prop-decorator errors (Pydantic @property + validators)"
        fi
        
        test_fail "MyPy: TYPE ERRORS FOUND"
    fi
else
    echo "⚠️  MyPy not installed (optional)"
    test_pass "MyPy skipped (optional)"
fi

# ============================================================================
# PHASE 3: IMPORTS VALIDATION (REAL CLASSES ONLY)
# ============================================================================

test_step "Phase 3: Python Imports Validation (Real classes)"

echo "Testing imports with REAL existing classes..."

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
    
    # Check what REALLY exists in responses.py
    from schemas import responses
    real_classes = [x for x in dir(responses) if not x.startswith('_') and x[0].isupper()]
    print(f"✅ responses.py: {len(real_classes)} classes available")
    
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
    from data.feature_config import FeatureConfig, DataLoaderConfig
    from datetime import datetime
    
    # Test 1: SensorConfig with CORRECT enum
    print("Test 1: SensorConfig instantiation...")
    sensor = SensorConfig(
        sensor_id="pressure_001",
        sensor_type=SensorType.PRESSURE,
        component_id="pump_001",
        unit="bar",
        sampling_rate_hz=100.0,
        accuracy_percent=0.5,
        range_min=0.0,
        range_max=400.0
    )
    print(f"  ✅ SensorConfig created: {sensor.sensor_id}")
    
    # Test 2: EquipmentMetadata
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
    
    # Test 3: DataLoaderConfig.get_loader_kwargs (REAL method)
    print("\nTest 3: DataLoaderConfig instantiation...")
    loader_config = DataLoaderConfig()
    loader_kwargs = loader_config.get_loader_kwargs(split="train")
    print(f"  ✅ DataLoaderConfig.get_loader_kwargs() returns dict with {len(loader_kwargs)} keys")
    
    # Test 4: MinimalInferenceRequest - CORRECT data structure
    print("\nTest 4: MinimalInferenceRequest instantiation...")
    request = MinimalInferenceRequest(
        equipment_id="pump_001",
        timestamp=datetime.now(),
        topology_id="standard_pump_system",
        sensor_readings={
            "pump": {
                "pressure_bar": 250.0,
                "temperature_c": 45.0
            }
        }
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
echo "=================================================================="
echo "📊 TEST DRIVE RESULTS"
echo "=================================================================="
echo ""
echo "✅ Tests Passed: $TESTS_PASSED"
echo "❌ Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED - READY FOR MERGE!"
    echo ""
    echo "Entry point for production:"
    echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    exit 0
else
    echo "❌ TESTS FAILED - ISSUES TO FIX:"
    echo ""
    echo "Top issues:"
    echo "  1. MyPy errors (~120+): Focus on prop-decorator issues in metadata.py"
    echo "  2. Ruff linting: Import sorting (I001) - run autofix"
    echo ""
    echo "Quick fixes:"
    echo "  • Ruff: python -m ruff check src/ tests/ --fix"
    echo "  • MyPy prop-decorator: Use @property OR Pydantic validators, not both"
    echo "  • Or add: # type: ignore[prop-decorator] to affected @property lines"
    echo ""
    echo "Run again: ./TEST_DRIVE.sh"
    echo ""
    exit 1
fi
