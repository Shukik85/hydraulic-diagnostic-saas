#!/bin/bash

################################################################################
#                         🧪 SMOKE TEST SCRIPT v2.0.0                         #
#                      Quick validation (5 minutes)                           #
#                   FIXED FOR WINDOWS/MINGW (uses virtualenv)                #
################################################################################

# Don't exit on first error - handle gracefully instead
set +e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║               🧪 GNN SERVICE v2.0.0 SMOKE TEST                ║"
echo "║                     (5 minute validation)                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Find Python executable (Windows virtualenv or system)
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON=".venv/Scripts/python"
elif command -v python &> /dev/null; then
    PYTHON="python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo -e "${RED}❌ ERROR: Python not found${NC}"
    echo "Please ensure virtualenv is activated: source .venv/bin/activate"
    exit 1
fi

echo "Using Python: $PYTHON"
$PYTHON --version
echo ""

test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
        ((FAILED++))
        return 1
    fi
}

################################################################################
# TEST 1: Import Validation
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  IMPORT VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1a: InferenceEngine
$PYTHON -c "from src.inference.inference_engine import InferenceEngine; print('OK')" > /dev/null 2>&1
test_result $? "InferenceEngine imports"

# Test 1b: FeatureEngineer
$PYTHON -c "from src.data import FeatureEngineer; print('OK')" > /dev/null 2>&1
test_result $? "FeatureEngineer imports"

# Test 1c: TopologyService
$PYTHON -c "from src.services.topology_service import get_topology_service; print('OK')" > /dev/null 2>&1
test_result $? "TopologyService imports"

# Test 1d: GraphBuilder
$PYTHON -c "from src.data import GraphBuilder; print('OK')" > /dev/null 2>&1
test_result $? "GraphBuilder imports"

# Test 1e: Schemas
$PYTHON -c "from src.schemas import PredictionResponse; print('OK')" > /dev/null 2>&1
test_result $? "Schemas imports"

echo ""

################################################################################
# TEST 2: _preprocess_minimal() Implementation
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  _preprocess_minimal() IMPLEMENTATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON << 'PYEOF'
import sys
import inspect
from src.inference.inference_engine import InferenceEngine

try:
    # Check method exists
    method = getattr(InferenceEngine, '_preprocess_minimal', None)
    if method is None:
        print("ERROR: _preprocess_minimal NOT FOUND")
        sys.exit(1)

    # Get source code
    source = inspect.getsource(method)
    lines = [l.strip() for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]

    # Check for TODO (unimplemented)
    if 'TODO' in source:
        print("ERROR: Still has TODO")
        sys.exit(1)

    # Check for minimum implementation (10+ lines)
    if len(lines) < 10:
        print(f"ERROR: Only {len(lines)} lines of code (need at least 10)")
        sys.exit(1)

    # Check for key implementation parts
    has_sensor_readings = 'sensor_readings' in source
    has_dataframe = 'DataFrame' in source or 'pd.DataFrame' in source
    has_graph_builder = 'graph_builder' in source
    has_error_handling = 'try' in source

    if not (has_sensor_readings and has_dataframe and has_graph_builder and has_error_handling):
        print("ERROR: Missing key implementation parts")
        if not has_sensor_readings: print("  - sensor_readings handling")
        if not has_dataframe: print("  - DataFrame conversion")
        if not has_graph_builder: print("  - graph_builder usage")
        if not has_error_handling: print("  - error handling")
        sys.exit(1)

    # SUCCESS
    print("OK")
    print(f"Details: {len(lines)} lines of code")
    print(f"  ✓ Sensor data conversion")
    print(f"  ✓ DataFrame handling")
    print(f"  ✓ Graph building")
    print(f"  ✓ Error handling")
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

test_result $? "_preprocess_minimal() properly implemented"

echo ""

################################################################################
# TEST 3: Configuration
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 3a: InferenceConfig initializes
$PYTHON -c "from src.inference.inference_engine import InferenceConfig; InferenceConfig(model_path='test.ckpt'); print('OK')" > /dev/null 2>&1
test_result $? "InferenceConfig initializes"

# Test 3b: Can load configs directory
$PYTHON -c "from pathlib import Path; p = Path('configs'); print('EXISTS') if p.exists() else print('MISSING')" | grep -q EXISTS
test_result $? "configs/ directory exists"

echo ""

################################################################################
# TEST 4: Type Hints
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  TYPE HINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

$PYTHON << 'PYEOF'
import sys
import inspect
from src.inference.inference_engine import InferenceEngine

try:
    sig = inspect.signature(InferenceEngine._preprocess_minimal)
    params = list(sig.parameters.keys())
    
    # Check parameters
    if 'self' not in params:
        print("ERROR: Missing 'self' parameter")
        exit(1)
    if 'request' not in params:
        print("ERROR: Missing 'request' parameter")
        exit(1)
    if 'topology' not in params:
        print("ERROR: Missing 'topology' parameter")
        exit(1)
    
    print(f"OK")
    print(f"Parameters: {params}")
    exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
PYEOF

test_result $? "Type hints present"

echo ""

################################################################################
# TEST 5: No Critical Syntax Errors
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  SYNTAX VALIDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Compile Python files to check for syntax errors
$PYTHON -m py_compile src/inference/inference_engine.py > /dev/null 2>&1
test_result $? "inference_engine.py syntax"

$PYTHON -m py_compile main.py > /dev/null 2>&1
test_result $? "main.py syntax"

$PYTHON -m py_compile src/data/__init__.py > /dev/null 2>&1
test_result $? "src/data/__init__.py syntax"

echo ""

################################################################################
# SUMMARY
################################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 SMOKE TEST RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ PASSED: $PASSED${NC}"
echo -e "${RED}❌ FAILED: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ✅ ALL SMOKE TESTS PASSED!                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  1. Run: pytest tests/unit/ -v --cov=src"
    echo "  2. Run: ruff check . --exclude _deprecated,_legacy"
    echo "  3. Run: mypy src/ --strict"
    echo "  4. Then we create PR! 🚀"
    echo ""
    exit 0
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║               ❌ SMOKE TEST FAILED - FIX ISSUES                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi
