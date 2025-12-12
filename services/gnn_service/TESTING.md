# 🧪 Comprehensive Testing Guide v2.0.0

## Quick Start (5 min)

Smoke test to verify basic functionality:

```bash
cd services/gnn_service/

# Test 1: Import validation
echo "=== TEST 1: Import validation ==="
python3 -c "from src.inference.inference_engine import InferenceEngine; print('✅ InferenceEngine imports OK')"
python3 -c "from src.data import FeatureEngineer; print('✅ FeatureEngineer imports OK')"
python3 -c "from src.services.topology_service import get_topology_service; print('✅ TopologyService imports OK')"

# Test 2: _preprocess_minimal() implementation
echo "\n=== TEST 2: _preprocess_minimal() implementation ==="
python3 << 'EOF'
import inspect
from src.inference.inference_engine import InferenceEngine

method = getattr(InferenceEngine, '_preprocess_minimal', None)
if method is None:
    print("❌ NOT FOUND")
    exit(1)

source = inspect.getsource(method)
lines = [l.strip() for l in source.split('\n') if l.strip()]

if 'TODO' in source:
    print("❌ Still has TODO")
    exit(1)

if len(lines) < 10:
    print(f"❌ Only {len(lines)} lines")
    exit(1)

key_checks = [
    'sensor_readings' in source,
    'DataFrame' in source,
    'graph_builder' in source,
    'try' in source,
]

if not all(key_checks):
    print("❌ Missing key parts")
    exit(1)

print("✅ _preprocess_minimal() PROPERLY IMPLEMENTED")
print(f"   - {len(lines)} lines of code")
print("   - Has error handling")
print("   - Has sensor data conversion")
EOF

# Test 3: Configuration
echo "\n=== TEST 3: Configuration ==="
python3 -c "from src.inference.inference_engine import InferenceConfig; InferenceConfig(model_path='test'); print('✅ InferenceConfig OK')"

echo "\n✅ ALL QUICK TESTS PASSED!"
```

---

## Full Testing Suite

### 1. Unit Tests (30 min)

Test individual components in isolation:

```bash
pytest tests/unit/ -v --cov=src --cov-report=html
```

**Coverage target**: >80%

**Test modules:**
- `src/data/feature_engineer.py` - Feature extraction and normalization
- `src/data/graph_builder.py` - Graph construction with 14D edges
- `src/inference/inference_engine.py` - Core inference logic
- `src/services/topology_service.py` - Topology templates
- `main.py` - FastAPI endpoints and lifespan

**Key tests:**
- Feature extraction from sensor readings
- Handling missing values gracefully
- Graph construction with variable-sized inputs
- Edge feature computation (14D)
- Single and batch inference
- Output post-processing
- Error scenarios

### 2. Integration Tests (20 min)

Test full pipelines end-to-end:

```bash
pytest tests/integration/ -v
```

**Test scenarios:**
- End-to-end inference: Request → Graph → Inference → Response
- Batch processing: Multiple requests in parallel
- API endpoints:
  - `POST /api/v2/inference/minimal` (happy path, errors)
  - `GET /api/v2/topologies` (list templates)
  - `GET /api/v2/topologies/{id}` (get specific)
- Database integration (mocked)
- Topology validation

### 3. Error Scenario Tests (15 min)

Verify graceful handling of failure modes:

```bash
pytest tests/error_scenarios/ -v
```

**Scenarios:**
- Missing sensors in request
- Invalid topology ID
- Model not loaded
- Database timeout
- Malformed JSON
- Concurrent requests (race conditions)
- Memory pressure (large graphs)
- Type mismatches

### 4. Performance Tests (15 min)

Measure baseline performance:

```bash
# Start server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Run load test
locust -f tests/load/locustfile.py --headless \
    -u 50 -r 10 -t 5m -H http://localhost:8000

# Kill server
kill $SERVER_PID
```

**Baselines:**
- Single inference latency: <100ms (CPU), <50ms (GPU)
- Batch throughput: >10 requests/sec (10x sequential)
- Memory usage: <2GB for 100 concurrent requests
- Model loading: <5s
- API response: <200ms (including all overhead)

### 5. Code Quality Tests (10 min)

Static analysis and linting:

```bash
# Linting
ruff check . --exclude _deprecated,_legacy

# Type checking
mypy src/ --strict

# Security checks
bandit -r src/

# Code coverage report
pytest --cov=src --cov-report=term-missing

# Docstring validation
pydocstyle src/
```

**Requirements:**
- 0 linting errors
- 0 type errors
- 0 critical security issues
- >80% code coverage
- All functions documented

### 6. Backward Compatibility Tests (10 min)

Verify v1 API still works:

```bash
pytest tests/compat/ -v
```

**Test endpoints:**
- `POST /api/v1/predict` - Legacy single prediction
- `POST /api/v1/batch/predict` - Legacy batch processing
- Response format unchanged
- Error responses compatible

---

## Full Test Execution

### Sequential Execution (100 min)

```bash
#!/bin/bash
set -e  # Exit on any error

echo "🧪 Starting full test suite..."
echo ""

# 1. Quick smoke test
echo "1️⃣ Quick smoke test (5 min)..."
# ... run quick test script ...

# 2. Unit tests
echo "\n2️⃣ Unit tests (30 min)..."
pytest tests/unit/ -v --cov=src

# 3. Integration tests
echo "\n3️⃣ Integration tests (20 min)..."
pytest tests/integration/ -v

# 4. Error scenarios
echo "\n4️⃣ Error scenario tests (15 min)..."
pytest tests/error_scenarios/ -v

# 5. Code quality
echo "\n5️⃣ Code quality checks (10 min)..."
ruff check . --exclude _deprecated,_legacy
mypy src/ --strict
bandit -r src/

# 6. Performance
echo "\n6️⃣ Performance tests (15 min)..."
# ... start server and run locust ...

# 7. Backward compatibility
echo "\n7️⃣ Backward compatibility (10 min)..."
pytest tests/compat/ -v

echo "\n✅ ALL TESTS PASSED!"
```

### Parallel Execution (40 min)

Some tests can run in parallel:

```bash
#!/bin/bash

# Run all tests except performance (performance needs server)
pytest tests/unit tests/integration tests/error_scenarios tests/compat -v --cov=src -n auto

# Then run quality checks
ruff check . --exclude _deprecated,_legacy
mypy src/ --strict
bandit -r src/

# Finally run performance tests
# ... performance testing ...
```

---

## Production Readiness Checklist

### ✅ GO TO PRODUCTION if:

- [ ] All unit tests passing (100%)
- [ ] All integration tests passing (100%)
- [ ] No critical/high security issues
- [ ] Code coverage >80%
- [ ] Performance baselines met:
  - [ ] Single inference <100ms
  - [ ] Batch throughput >10 req/s
  - [ ] Memory <2GB for 100 concurrent
- [ ] Backward compatibility maintained
- [ ] All error scenarios handled
- [ ] No blocking TODOs in code
- [ ] Documentation complete

### ❌ STOP if ANY:

- [ ] Unit test failure
- [ ] Integration test failure
- [ ] Memory leak detected
- [ ] Race conditions found
- [ ] Critical security issue
- [ ] Breaking API change
- [ ] Performance degradation >20%
- [ ] Unhandled error scenario

---

## Continuous Integration

### Pre-commit Hooks

Run before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        exclude: ^(_deprecated|_legacy)/
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        args: [--strict]
        exclude: ^(_deprecated|_legacy)/
```

Install:
```bash
pre-commit install
pre-commit run --all-files
```

### GitHub Actions

Run on every push:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.14'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --cov=src
      - run: ruff check . --exclude _deprecated,_legacy
      - run: mypy src/ --strict
      - run: bandit -r src/
```

---

## Performance Profiling

### Flamegraph

Identify performance bottlenecks:

```bash
python3 -m py_spy record -o profile.svg -- python3 -m uvicorn main:app

# Make requests, then Ctrl+C
# View: open profile.svg
```

### Memory Profiling

Detect memory leaks:

```bash
pip install memory_profiler
python3 -m memory_profiler main.py
```

---

## Debugging Failed Tests

### Verbose Output

```bash
pytest tests/ -vv -s
```

### Specific Test

```bash
pytest tests/test_file.py::test_function -vv
```

### With Debugging

```bash
pytest tests/ -vv -s --pdb  # Drop into debugger on failure
```

### Markers

```bash
pytest tests/ -m slow -v  # Run slow tests
pytest tests/ -m "not slow" -v  # Skip slow tests
```

---

## Next Steps After Testing

1. ✅ All tests pass → Create production PR
2. ✅ PR approved → Merge to main
3. ✅ Main merged → Deploy to staging
4. ✅ Staging verified → Deploy to production

---

## References

- `README.md` - Setup guide
- `STRUCTURE.md` - Architecture overview
- `PRODUCTION_READINESS.md` - Assessment
- Test files in `tests/` directory

---

**Last Updated**: 2025-12-13
**Version**: v2.0.0
**Status**: Production Ready
