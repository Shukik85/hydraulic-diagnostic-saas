# Testing Guide

This document describes the testing approach and instructions for the Hydraulic Diagnostic SaaS project.

## Overview

The project uses:
- **Backend**: pytest with pytest-django for Django unit and integration tests
- **Frontend**: Vitest with @vue/test-utils for Vue component tests
- **CI/CD**: GitHub Actions for automated testing

---

## Backend Testing

### Test Structure

```
backend/
├── tests/
│   ├── __init__.py
│   ├── test_models.py        # Django model tests
│   ├── test_serializers.py   # DRF serializer tests
│   └── test_viewsets.py      # API endpoint integration tests
├── conftest.py               # Pytest fixtures
└── pytest.ini                # Pytest configuration
```

### Running Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=apps --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests with verbose output
pytest -v

# Run only database tests
pytest -m django_db
```

### Writing Backend Tests

**Example: Model Test**
```python
import pytest
from apps.diagnostics.models import Equipment

@pytest.mark.django_db
class TestEquipmentModel:
    def test_equipment_creation(self):
        equipment = Equipment.objects.create(
            name="Test Pump",
            equipment_type="pump"
        )
        assert equipment.name == "Test Pump"
```

**Example: ViewSet Test**
```python
import pytest
from rest_framework.test import APIClient

@pytest.mark.django_db
class TestEquipmentViewSet:
    def test_list_equipment(self, api_client):
        response = api_client.get('/api/equipment/')
        assert response.status_code == 200
```

---

## Frontend Testing

### Test Structure

```
frontend/
└── tests/
    └── components/
        └── EquipmentCard.test.js
```

### Running Frontend Tests

```bash
cd frontend

# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode (development)
npm test -- --watch

# Run specific test file
npm test EquipmentCard.test.js
```

### Writing Frontend Tests

**Example: Component Test**
```javascript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EquipmentCard from '@/components/EquipmentCard.vue'

describe('EquipmentCard.vue', () => {
  it('renders equipment name', () => {
    const wrapper = mount(EquipmentCard, {
      props: {
        equipment: { id: 1, name: 'Pump A' }
      }
    })
    expect(wrapper.text()).toContain('Pump A')
  })
})
```

---

## CI/CD Pipeline

### Automated Testing

GitHub Actions automatically runs tests on:
- Push to `master` or `develop` branches
- Pull requests targeting `master` or `develop`

### Pipeline Jobs

1. **test-backend**: Runs pytest with PostgreSQL service
2. **test-frontend**: Runs Vitest with coverage
3. **lint-backend**: Runs flake8 linting
4. **lint-frontend**: Runs ESLint

### Viewing CI Results

- Navigate to **Actions** tab in GitHub
- Select a workflow run to see detailed logs
- Coverage reports are uploaded to Codecov (if configured)

### Local CI Testing

To run tests as CI would:

```bash
# Backend
cd backend
pip install -r requirements.txt
pytest --cov=apps --cov-report=term

# Frontend
cd frontend
npm ci
npm test -- --run --coverage
```

---

## Test Coverage Goals

- **Backend**: Minimum 80% code coverage
- **Frontend**: Minimum 70% code coverage
- **Critical paths**: 100% coverage (authentication, payments, data integrity)

---

## Best Practices

### Backend
- Use factories (via `conftest.py`) for test data
- Mark database tests with `@pytest.mark.django_db`
- Test both success and error scenarios
- Mock external services and API calls

### Frontend
- Test component behavior, not implementation
- Use data-testid attributes for reliable selectors
- Test user interactions and event emissions
- Mock API calls with MSW or similar tools

### General
- Write descriptive test names: `test_<feature>_<scenario>`
- Keep tests independent and isolated
- Run tests before committing code
- Update tests when requirements change

---

## Troubleshooting

### Backend Issues

**Database connection errors**
```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/test_db
```

**Import errors**
```bash
export DJANGO_SETTINGS_MODULE=core.settings
```

### Frontend Issues

**Module not found**
```bash
npm install
```

**Test timeouts**
```javascript
// In test file
it('long test', { timeout: 10000 }, () => { ... })
```

---

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-django documentation](https://pytest-django.readthedocs.io/)
- [Vitest documentation](https://vitest.dev/)
- [Vue Test Utils](https://test-utils.vuejs.org/)
- [GitHub Actions documentation](https://docs.github.com/actions)
