# 🧪 Testing Roadmap - Local Validation

**Goal**: Run comprehensive tests BEFORE creating PR
**Timeline**: ~100 minutes total
**Status**: READY TO START

---

## ❓ Prerequisites

```bash
cd services/gnn_service/

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Verify Python
python3 --version  # Should be 3.14+

# Verify pip packages installed
pip list | grep torch
pip list | grep pytest
```

---

## 🧪 TEST EXECUTION PLAN

### PHASE 1: Quick Smoke Test (5 minutes) ⚡️

**STATUS**: Ready Now

```bash
# Make script executable
chmod +x tests/smoke_test.sh

# Run smoke test
bash tests/smoke_test.sh
```

**Expected Output**:
```
╔════════════════════════════════════════════════════════════════╗
║               🧪 GNN SERVICE v2.0.0 SMOKE TEST                ║
╚════════════════════════════════════════════════════════════════╝

1️⃣  IMPORT VALIDATION
✅ PASS: InferenceEngine imports
✅ PASS: FeatureEngineer imports
✅ PASS: TopologyService imports
✅ PASS: GraphBuilder imports
✅ PASS: Schemas imports

2️⃣  _preprocess_minimal() IMPLEMENTATION
✅ PASS: _preprocess_minimal() properly implemented
  Details: 42 lines of code
  ✓ Sensor data conversion
  ✓ DataFrame handling
  ✓ Graph building
  ✓ Error handling

3️⃣  CONFIGURATION
✅ PASS: InferenceConfig initializes
✅ PASS: configs/ directory exists

4️⃣  TYPE HINTS
✅ PASS: Type hints present
Parameters: ['self', 'request', 'topology']

5️⃣  SYNTAX VALIDATION
✅ PASS: inference_engine.py syntax
✅ PASS: main.py syntax
✅ PASS: src/data/__init__.py syntax

📊 SMOKE TEST RESULTS
✅ PASSED: 13
❌ FAILED: 0

╔════════════════════════════════════════════════════════════════╗
║                  ✅ ALL SMOKE TESTS PASSED!                   ║
╚════════════════════════════════════════════════════════════════╝
```

**Go/No-Go**: ❌ If ANY test fails, STOP and fix

---

### PHASE 2: Unit Tests (30 minutes) ❔️⃣

**STATUS**: After Smoke Test Passes

```bash
# Run unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=html
```

**Expected Output**:
```
============================= test session starts ==============================
platform linux -- Python 3.14.0
collected 25 items

tests/unit/test_feature_engineer.py PASSED                             [ 4%]
tests/unit/test_graph_builder.py PASSED                                [ 8%]
tests/unit/test_inference_engine.py PASSED                             [12%]
... (more tests)

============================== coverage ========================================
Name                              Stmts   Miss  Cover
───────────────────────────────────────
src/data/feature_engineer.py          120     8   93%
src/data/graph_builder.py             150    10   93%
src/inference/inference_engine.py     200    15   92%
src/models/gnn.py                      95     5   95%
src/services/topology_service.py       80     3   96%
───────────────────────────────────────
TOTAL                                 1000   50   95%

============================== 25 passed in 12.34s ==============================
```

**Go/No-Go**: ✅ All tests pass + >80% coverage

---

### PHASE 3: Integration Tests (20 minutes) ❕️⃣

**STATUS**: After Unit Tests Pass

```bash
# Run integration tests
pytest tests/integration/ -v
```

**Expected Output**:
```
============================= test session starts ==============================
platform linux -- Python 3.14.0
collected 12 items

tests/integration/test_e2e_inference.py::test_minimal_prediction PASSED [8%]
tests/integration/test_e2e_inference.py::test_batch_processing PASSED  [16%]
tests/integration/test_api_endpoints.py::test_health_check PASSED      [25%]
tests/integration/test_api_endpoints.py::test_ready_check PASSED       [33%]
tests/integration/test_api_endpoints.py::test_minimal_inference PASSED [41%]
tests/integration/test_topology.py::test_topology_templates PASSED     [50%]
... (more tests)

============================== 12 passed in 8.45s ==============================
```

**Go/No-Go**: ✅ All integration tests pass

---

### PHASE 4: Error Scenario Tests (15 minutes) ❖️⃣

**STATUS**: After Integration Tests Pass

```bash
# Run error scenario tests
pytest tests/error_scenarios/ -v
```

**Expected Output**:
```
============================= test session starts ==============================
platform linux -- Python 3.14.0
collected 8 items

tests/error_scenarios/test_missing_sensors.py PASSED                   [12%]
tests/error_scenarios/test_invalid_topology.py PASSED                  [25%]
tests/error_scenarios/test_malformed_request.py PASSED                 [37%]
tests/error_scenarios/test_concurrent_requests.py PASSED               [50%]
... (more tests)

============================== 8 passed in 6.23s ==============================
```

**Go/No-Go**: ✅ All errors handled gracefully

---

### PHASE 5: Code Quality Checks (10 minutes) ❗️⃣

**STATUS**: After Functional Tests Pass

```bash
# Linting
echo "=== Running Ruff Linter ==="
ruff check . --exclude _deprecated,_legacy
echo "Status: $?"

# Type checking
echo "\n=== Running MyPy Type Checker ==="
mypy src/ --strict
echo "Status: $?"

# Security
echo "\n=== Running Bandit Security Scan ==="
bandit -r src/
echo "Status: $?"
```

**Expected Output**:
```
=== Running Ruff Linter ===
Status: 0  (no errors)

=== Running MyPy Type Checker ===
Status: 0  (no errors)

=== Running Bandit Security Scan ===
No issues identified
Status: 0
```

**Go/No-Go**: ✅ 0 linting errors + 0 type errors + 0 security issues

---

### PHASE 6: Backward Compatibility (10 minutes) ❘️⃣

**STATUS**: After Code Quality Passes

```bash
# Run compatibility tests
pytest tests/compat/ -v
```

**Expected Output**:
```
============================= test session starts ==============================
platform linux -- Python 3.14.0
collected 4 items

tests/compat/test_v1_api.py::test_v1_predict PASSED                    [25%]
tests/compat/test_v1_api.py::test_v1_batch_predict PASSED              [50%]
tests/compat/test_v1_api.py::test_response_format_unchanged PASSED     [75%]

============================== 4 passed in 3.21s ==============================
```

**Go/No-Go**: ✅ v1 API still works

---

### PHASE 7: Optional - Performance Baseline (15 minutes) 🔗

**STATUS**: Optional (for production deployment)

```bash
# Start server in background
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Run load test
locust -f tests/load/locustfile.py --headless \
    -u 10 -r 2 -t 2m -H http://localhost:8000

# Kill server
kill $SERVER_PID
```

**Expected Output**:
```
Response time 50th percentile: 45ms
Response time 95th percentile: 120ms
Requests/sec: 8.5
Failures: 0%
```

---

## 🎉 FULL TEST SUMMARY

After all phases complete:

```bash
echo "✅ ALL TESTS PASSED"
echo "✅ Code coverage: >80%"
echo "✅ No linting errors"
echo "✅ No type errors"
echo "✅ No security issues"
echo "✅ Backward compatible"
echo ""
echo "🚀 READY FOR PR!"
```

---

## 📑 Production Readiness Checklist

Before creating PR, verify:

- [ ] Smoke tests: PASS (13/13)
- [ ] Unit tests: PASS (>80% coverage)
- [ ] Integration tests: PASS (all scenarios)
- [ ] Error scenarios: PASS (graceful handling)
- [ ] Code quality: PASS (0 errors)
- [ ] Backward compatibility: PASS (v1 API works)
- [ ] No TODO comments in code
- [ ] Documentation complete
- [ ] Git history clean (22 commits)

---

## 🚀 Next Steps After All Tests Pass

1. **Create PR**
   ```bash
   git push origin feature/gnn-service-production-ready
   # Go to GitHub and create PR
   ```

2. **Link to Testing Results**
   - Attach smoke test output
   - Mention coverage percentage
   - Note any performance metrics

3. **Code Review**
   - Request review from team
   - Address any feedback
   - Merge to main

4. **Deploy to Staging**
   - Pull main branch
   - Run deployment tests
   - Verify in staging environment

5. **Production Deployment**
   - Blue-green deployment
   - Health check verification
   - Monitor metrics

---

## ⚠️ STOPPING CRITERIA

**Stop and fix if ANY of these occur:**

- ❌ Smoke test fails
- ❌ Unit test fails (<80% coverage)
- ❌ Integration test fails
- ❌ Linting errors detected
- ❌ Type errors found
- ❌ Security issues found
- ❌ Backward compatibility broken
- ❌ Unhandled error scenario

---

## 📞 Support

If tests fail:

1. Check error message
2. Review relevant code
3. Fix the issue
4. Re-run that test phase
5. Only proceed when GREEN

---

**Estimated Total Time**: ~100 minutes
**Current Phase**: Ready to Start (Phase 1)
🚀
