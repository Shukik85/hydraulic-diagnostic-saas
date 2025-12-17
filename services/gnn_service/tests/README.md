# GNN Service Test Suite

Comprehensive testing for hydraulic diagnostic GNN service.

## Directory Structure

```
tests/
├── unit/                           # Unit tests
│   ├── test_losses.py             # Loss functions
│   ├── test_metrics.py            # Metrics
│   ├── test_graph_reconstructor.py # Graph reconstruction
│   ├── test_edge_in_dim.py        # Edge dimensions
│   └── test_training/             # Training components
│       ├── test_dataloader_temporal.py
│       ├── test_lightning_module.py
│       └── test_imputation_engine.py
├── integration/                    # Integration tests  
│   └── test_full_pipeline.py      # End-to-end pipeline
├── conftest.py                     # Shared fixtures
├── pytest.ini                      # Pytest configuration
└── README.md                       # This file
```

## Running Tests

### All tests
```bash
pytest tests/
```

### Only unit tests
```bash
pytest tests/unit/ -m unit
```

### Only integration tests
```bash
pytest tests/integration/ -m integration
```

### Specific test file
```bash
pytest tests/unit/test_losses.py -v
```

### With coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Specific test
```bash
pytest tests/unit/test_losses.py::TestFocalLoss::test_focal_loss_basic -v
```

### Parallel execution
```bash
pytest tests/ -n auto
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests with timeout > 5s
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.smoke` - Smoke tests

### Using markers
```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run only GPU tests
pytest tests/ -m gpu

# Run everything except slow GPU tests
pytest tests/ -m "not (slow and gpu)"
```

## Fixtures

### Data Fixtures

- `sample_graph` - Single graph with 10 nodes, 15 edges
- `sample_graphs` - List of 5 sample graphs
- `sample_batch` - Batched graphs
- `sample_temporal_sequence` - Temporal sequence of 12 snapshots

### Mock Fixtures

- `mock_timescale_connector` - Mock TimescaleDB connector
- `mock_feature_engineer` - Mock feature extraction
- `mock_graph_topology` - Mock graph topology

### Path Fixtures

- `tmp_checkpoint_dir` - Temporary checkpoint directory
- `tmp_log_dir` - Temporary log directory

### Device Fixtures

- `device` - GPU if available, else CPU
- `cpu_device` - Always CPU

## Coverage

Current coverage targets:

- `src/training/losses.py` - 95%+
- `src/training/metrics.py` - 95%+
- `src/training/lightning_module.py` - 90%+
- `src/training/dataloader_temporal.py` - 85%+
- `src/training/imputation_engine.py` - 80%+
- `src/training/graph_reconstructor.py` - 75%+

View HTML coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## CI/CD Integration

### GitHub Actions

Tests run on every commit:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src
```

## Best Practices

1. **Use fixtures** - Don't create test data inline
2. **One assertion** per test when possible
3. **Descriptive names** - Test name should describe what's tested
4. **Parametrize** - Use `@pytest.mark.parametrize` for multiple inputs
5. **Mock external deps** - Mock DB, APIs, file I/O
6. **Isolate tests** - Tests should not depend on each other
7. **Use markers** - Tag tests with appropriate markers

## Troubleshooting

### Tests fail with CUDA errors
```bash
# Force CPU
pytest tests/ --cpu
```

### Tests timeout
```bash
# Increase timeout
pytest tests/ --timeout=600
```

### Flaky tests
```bash
# Run multiple times
pytest tests/ --count=10
```

### See debug output
```bash
pytest tests/ -vv -s
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD)
2. Place tests in appropriate directory
3. Use existing fixtures when possible
4. Add test markers
5. Document complex test logic
6. Aim for >80% coverage

Example test:

```python
import pytest

class TestMyFeature:
    """Test my new feature."""
    
    @pytest.fixture
    def my_fixture(self):
        return SomeObject()
    
    def test_basic_functionality(self, my_fixture):
        """Test basic functionality works."""
        result = my_fixture.do_something()
        assert result == expected_value
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test with multiple inputs."""
        assert compute(input) == expected
```
