# Tests

Unit and integration tests for GNN service.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Test Structure

- `test_model.py` - Model architecture tests
- `test_inference.py` - Inference engine tests
- `test_rag.py` - RAG service tests
- `test_api.py` - API endpoint tests (TODO)

## Coverage Target

>= 90% code coverage for production deployment.
