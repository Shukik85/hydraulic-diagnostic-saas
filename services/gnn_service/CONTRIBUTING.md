# Contributing to GNN Service

Thank you for your interest in contributing! This guide will help you set up your development environment and understand our workflow.

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.11+ (3.12 recommended)
- Git
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas/services/gnn_service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov pytest-asyncio
pip install black ruff mypy
pip install pre-commit

# Setup pre-commit hooks
pre-commit install
```

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report

# Specific test file
pytest tests/unit/test_model_manager.py -v

# Specific test
pytest tests/unit/test_model_manager.py::TestModelLoading::test_load_model_success -v
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests (requires CUDA)
pytest -m gpu
```

---

## 🎨 Code Style

### Formatting with Black

```bash
# Check formatting
black --check src/ tests/

# Auto-format
black src/ tests/
```

### Linting with Ruff

```bash
# Check linting
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

### Type Checking with MyPy

```bash
# Check types
mypy src/ --ignore-missing-imports
```

### Pre-commit Hooks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run on staged files (automatic on commit)
git commit -m "Your message"
```

---

## 📐 Code Guidelines

### General Principles

- **Type hints:** Use type hints for all function signatures
- **Docstrings:** Document all public classes and functions
- **Testing:** Write tests for all new features
- **Coverage:** Maintain >90% test coverage
- **Formatting:** Follow Black code style
- **Imports:** Use isort (automatic with ruff)

### Example

```python
"""Module description.

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field


class ExampleClass:
    """Example class with proper documentation.
    
    Attributes:
        name: Name of the example
        value: Numeric value
    
    Examples:
        >>> example = ExampleClass(name="test", value=42)
        >>> example.process()
        42
    """
    
    def __init__(self, name: str, value: int):
        """Initialize example.
        
        Args:
            name: Name parameter
            value: Value parameter
        """
        self.name = name
        self.value = value
    
    def process(self) -> int:
        """Process the value.
        
        Returns:
            processed_value: The processed value
        
        Examples:
            >>> example.process()
            42
        """
        return self.value
```

---

## 🔄 Git Workflow

### Branch Naming

- Feature: `feature/description`
- Bugfix: `fix/description`
- Hotfix: `hotfix/description`
- Docs: `docs/description`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Testing
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes
- `chore`: Maintenance

**Examples:**

```bash
feat(inference): add batch prediction support

- Implement batch processing in InferenceEngine
- Add BatchPredictionRequest/Response schemas
- Add /predict/batch endpoint
- Add integration tests

Closes #96
```

### Pull Request Process

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Push to GitHub:**
   ```bash
   git push origin feature/my-feature
   ```

4. **Create Pull Request:**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in description
   - Request review

5. **CI checks must pass:**
   - ✅ All tests pass
   - ✅ Coverage ≥90%
   - ✅ Linting passes
   - ✅ Type checking passes

6. **Merge after approval**

---

## 🚀 CI/CD

### GitHub Actions

Our CI/CD pipeline automatically:

1. **Runs tests** on Python 3.11, 3.12, 3.13
2. **Checks coverage** (must be ≥90%)
3. **Runs linting** (ruff, black, mypy)
4. **Uploads coverage** to Codecov

### Workflow Files

- `.github/workflows/gnn-service-ci.yml` - Main CI workflow

### Local CI Simulation

```bash
# Run the same checks as CI
pytest --cov=src --cov-report=term-missing --cov-fail-under=90
ruff check src/ tests/
black --check src/ tests/
mypy src/ --ignore-missing-imports
```

---

## 📦 Dependencies

### Adding New Dependencies

1. **Add to requirements.txt:**
   ```bash
   echo "new-package>=1.0.0" >> requirements.txt
   ```

2. **Install:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Freeze versions (optional):**
   ```bash
   pip freeze > requirements-lock.txt
   ```

---

## 🐛 Troubleshooting

### Pre-commit Hook Issues

```bash
# Update pre-commit
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install
```

### Test Failures

```bash
# Run with verbose output
pytest -vv --tb=long

# Run single test with print statements
pytest tests/unit/test_file.py::test_name -s

# Clear pytest cache
rm -rf .pytest_cache
```

### Import Errors

```bash
# Ensure you're in project root
cd services/gnn_service

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pytest Documentation](https://docs.pytest.org/)

---

## 💬 Getting Help

- **Issues:** Open an issue on GitHub
- **Discussions:** Use GitHub Discussions
- **Email:** shukik85@ya.ru

---

**Thank you for contributing!** 🎉
