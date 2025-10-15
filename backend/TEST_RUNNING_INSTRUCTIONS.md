# Backend Testing Instructions

## Prerequisites

```bash
pip install pytest pytest-django pytest-cov djangorestframework
```

## Running Tests

### Run all tests
```bash
cd backend
pytest
```

### Run with coverage
```bash
pytest --cov=apps --cov-report=html
```

### Run specific test modules
```bash
pytest tests/test_models.py
pytest tests/test_serializers.py
pytest tests/test_viewsets.py
```

### Run tests by marker
```bash
pytest -m django_db  # Run only database tests
pytest -m slow       # Run only slow tests
```

### Verbose output
```bash
pytest -v
```

## Test Structure

- `tests/test_models.py` - Unit tests for Django models
- `tests/test_serializers.py` - Unit tests for DRF serializers
- `tests/test_viewsets.py` - Integration tests for API viewsets
- `conftest.py` - Shared pytest fixtures and configuration
- `pytest.ini` - Pytest configuration

## Writing New Tests

1. Create test files with `test_` prefix
2. Use fixtures from `conftest.py` for common test data
3. Mark database tests with `@pytest.mark.django_db`
4. Follow naming convention: `test_<feature>_<scenario>`

## CI/CD Integration

Tests are automatically run via GitHub Actions on push and pull request.
