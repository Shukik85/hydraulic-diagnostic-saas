---

# Testing Guide (Revised)

## Overview

### Testing Philosophy
We follow the **Testing Pyramid** strategy:
- **70% Unit Tests** - Fast, isolated tests for functions, components, models
- **20% Integration Tests** - API endpoints, service integrations
- **10% E2E Tests** - Critical user workflows

### Quality Gates
- Backend coverage ≥ 80%
- Frontend coverage ≥ 70%
- All tests must pass in CI before deployment
- E2E tests for core user journeys

---

## Backend Testing (Enhanced)

### Test Environment Setup

**conftest.py** (crucial for real-world testing):

```python
# backend/conftest.py
import pytest
from django.test import Client
from rest_framework.test import APIClient
from apps.diagnostics.factories import EquipmentFactory, DiagnosticReportFactory, UserFactory

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def authenticated_client():
    client = APIClient()
    user = UserFactory()
    client.force_authenticate(user=user)
    return client

@pytest.fixture
def sample_hydraulic_system():
    """Creates a complete test system with related equipment"""
    system = EquipmentFactory(
        name="Main Hydraulic System",
        system_type="complex",
        pressure_rating=350
    )
    # Create related components
    EquipmentFactory.create_batch(3, parent_system=system)
    return system
```

### Realistic Test Examples

**Model Tests with Business Logic:**

```python
# tests/test_models.py
@pytest.mark.django_db
class TestDiagnosticReport:
    def test_report_risk_calculation(self):
        report = DiagnosticReportFactory(
            pressure_anomalies=5,
            temperature_anomalies=2,
            contamination_level=0.8
        )

        # Test business logic - risk calculation
        expected_risk = "HIGH"
        assert report.calculate_risk_level() == expected_risk

    def test_report_status_flow(self):
        report = DiagnosticReportFactory(status="RUNNING")

        # Test valid status transition
        report.mark_completed()
        assert report.status == "COMPLETED"

        # Test invalid transition
        with pytest.raises(ValidationError):
            report.mark_failed()  # Cannot fail after completion
```

**API Integration Tests with Mocking:**

```python
# tests/test_viewsets.py
@pytest.mark.django_db
class TestDiagnosticViewSet:
    def test_run_diagnostic_authenticated(self, authenticated_client, mocker):
        # Mock external hydraulic analysis service
        mock_analysis = mocker.patch('apps.diagnostics.services.HydraulicAnalysisAPI.run')
        mock_analysis.return_value = {
            "anomalies": 3,
            "severity": "HIGH",
            "recommendations": ["Check pump seals", "Monitor pressure"]
        }

        equipment = EquipmentFactory()
        payload = {
            "equipment_id": equipment.id,
            "test_parameters": {"pressure": 350, "duration": 30}
        }

        response = authenticated_client.post('/api/diagnostics/run/', payload, format='json')

        assert response.status_code == 201
        assert response.data['severity'] == "HIGH"
        mock_analysis.assert_called_once()

    def test_diagnostic_performance_benchmark(self, benchmark, authenticated_client):
        # Performance test for diagnostic generation
        equipment = EquipmentFactory()

        def run_diagnostic():
            return authenticated_client.post('/api/diagnostics/quick-scan/',
                                           {"equipment_id": equipment.id})

        result = benchmark(run_diagnostic)
        assert result.status_code == 201
        assert benchmark.stats['mean'] < 1.0  # Must complete under 1 second
```

---

## Frontend Testing (Enhanced)

### Test Utilities Setup

**Custom test utilities:**

```javascript
// frontend/tests/setup.js
import { config } from '@vue/test-utils';
import { vi } from 'vitest';

// Global mocks
vi.mock('@/services/api');
vi.mock('@/services/websockets');

// Custom data-testid plugin
config.global.plugins.push({
  install(app) {
    app.config.globalProperties.$testId = id => `[data-testid="${id}"]`;
  },
});
```

### Realistic Component Tests

**EquipmentDashboard.vue Test:**

```javascript
// tests/components/EquipmentDashboard.test.js
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import EquipmentDashboard from '@/components/EquipmentDashboard.vue';
import api from '@/services/api';

describe('EquipmentDashboard.vue', () => {
  let mockEquipmentData;

  beforeEach(() => {
    mockEquipmentData = [
      { id: 1, name: 'Hydraulic Pump A', status: 'online', pressure: 350 },
      { id: 2, name: 'Control Valve B', status: 'offline', pressure: 0 },
    ];

    vi.clearAllMocks();
    api.getEquipment.mockResolvedValue({ data: mockEquipmentData });
  });

  it('loads and displays equipment list with real-time data', async () => {
    const wrapper = mount(EquipmentDashboard);

    // Initial loading state
    expect(wrapper.find('[data-testid="loading"]').exists()).toBe(true);

    await flushPromises();

    // Verify data is displayed
    expect(wrapper.text()).toContain('Hydraulic Pump A');
    expect(wrapper.text()).toContain('350 psi');
    expect(wrapper.text()).toContain('offline');
  });

  it('filters equipment by status', async () => {
    const wrapper = mount(EquipmentDashboard);
    await flushPromises();

    // Test filter functionality
    await wrapper.find('[data-testid="filter-online"]').setValue(true);

    expect(wrapper.findAll('[data-testid="equipment-card"]')).toHaveLength(1);
    expect(wrapper.text()).not.toContain('Control Valve B');
  });

  it('handles real-time data updates via WebSocket', async () => {
    const wrapper = mount(EquipmentDashboard);
    await flushPromises();

    // Simulate WebSocket message
    const mockWebSocketMessage = {
      equipment_id: 1,
      pressure: 375,
      status: 'warning',
    };

    window.dispatchEvent(
      new MessageEvent('message', {
        data: JSON.stringify(mockWebSocketMessage),
      })
    );

    await flushPromises();

    // Verify UI updates
    expect(wrapper.text()).toContain('375 psi');
    expect(wrapper.find('[data-testid="status-warning"]').exists()).toBe(true);
  });

  it('shows error state when API fails', async () => {
    api.getEquipment.mockRejectedValue(new Error('Network error'));

    const wrapper = mount(EquipmentDashboard);
    await flushPromises();

    expect(wrapper.find('[data-testid="error-message"]').exists()).toBe(true);
    expect(wrapper.text()).toContain('Failed to load equipment');
  });
});
```

---

## End-to-End Testing (New Section)

### Playwright Setup

**playwright.config.js:**

```javascript
// frontend/playwright.config.js
module.exports = {
  use: {
    baseURL: process.env.TEST_URL || 'http://localhost:3000',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
};
```

### Critical User Journey Tests

**diagnostic-workflow.spec.js:**

```javascript
// tests/e2e/diagnostic-workflow.spec.js
import { test, expect } from '@playwright/test';

test.describe('Diagnostic Workflow', () => {
  test('complete diagnostic analysis flow', async ({ page }) => {
    // 1. Authentication
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'engineer@company.com');
    await page.fill('[data-testid="password"]', 'test123');
    await page.click('[data-testid="login-btn"]');

    await expect(page).toHaveURL(/\/dashboard/);

    // 2. Navigate to equipment
    await page.click('[data-testid="equipment-link"]');
    await expect(page.locator('[data-testid="equipment-list"]')).toBeVisible();

    // 3. Select equipment and run diagnostic
    await page.click('[data-testid="equipment-card"]:first-child');
    await page.click('[data-testid="run-diagnostic"]');

    // 4. Configure parameters
    await page.fill('[data-testid="pressure-input"]', '350');
    await page.selectOption('[data-testid="test-type"]', 'comprehensive');
    await page.click('[data-testid="start-analysis"]');

    // 5. Wait for results and verify
    await expect(page.locator('[data-testid="results-section"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('text=Analysis Complete')).toBeVisible();

    // 6. Generate report
    await page.click('[data-testid="generate-report"]');
    await expect(page).toHaveURL(/\/report\/.*/);

    // 7. Validate report content
    await expect(page.locator('text=Diagnostic Report')).toBeVisible();
    await expect(page.locator('text=Recommendations')).toBeVisible();
  });
});
```

---

## Performance Testing (New Section)

### Backend Performance Tests

```python
# tests/performance/test_diagnostic_performance.py
import pytest
from apps.diagnostics.services import ComplexDiagnosticEngine

@pytest.mark.performance
class TestDiagnosticPerformance:

    def test_complex_diagnostic_performance(self, benchmark, large_equipment_dataset):
        engine = ComplexDiagnosticEngine()

        def run_complex_analysis():
            return engine.analyze_system(large_equipment_dataset, depth="deep")

        result = benchmark(run_complex_analysis)

        # Performance assertions
        assert result is not None
        assert benchmark.stats['mean'] < 2.5  # Must complete under 2.5 seconds
        assert benchmark.stats['max'] < 5.0   # Never exceed 5 seconds

    def test_memory_usage_during_batch_processing(self):
        import tracemalloc

        tracemalloc.start()

        # Process large batch of diagnostics
        process_large_batch()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 100 * 1024 * 1024  # Peak memory < 100MB
```

---

## CI/CD Integration (Enhanced)

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: tester
          POSTGRES_PASSWORD: testpass
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest coverage pytest-django
      - name: Run tests with coverage
        run: |
          cd backend
          pytest --cov=apps --cov-report=xml --cov-report=html -v
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      - name: Run unit tests
        run: |
          cd frontend
          npm test -- --coverage --passWithNoTests
      - name: Run E2E tests
        run: |
          cd frontend
          npx playwright install
          npm run test:e2e
        env:
          TEST_URL: ${{ secrets.TEST_URL }}

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Run security audit
        run: |
          cd backend && pip-audit
          cd frontend && npm audit
```

---

## Best Practices (Enhanced)

### Backend Testing

- **Factory Pattern**: Use factory_boy for test data creation
- **Mock External Services**: Always mock APIs, email services, payment gateways
- **Database Isolation**: Use transaction rollbacks for test isolation
- **Performance Monitoring**: Benchmark critical paths regularly

### Frontend Testing

- **Test User Behavior**: Focus on what users do, not implementation details
- **Mock API Responses**: Use MSW (Mock Service Worker) for realistic API mocking
- **Accessibility Testing**: Include a11y checks in component tests
- **Visual Testing**: Use Percy or similar for UI regression testing

### E2E Testing

- **Test Critical Paths Only**: Focus on key user journeys
- **Use Realistic Data**: Mimic production data scenarios
- **Parallel Execution**: Run tests in parallel for speed
- **Cross-browser Testing**: Test on multiple browsers in CI

---

## Troubleshooting (Enhanced)

### Common Issues & Solutions

**Backend Database Conflicts:**

```bash
# Clear test database
python manage.py flush --settings=core.settings.test
# Recreate test schema
pytest --create-db
```

**Frontend Test Timeouts:**

```javascript
// Increase timeout for complex components
it('loads complex data', { timeout: 10000 }, async () => {
  // test implementation
});

// Use waitFor for async updates
await waitFor(() => {
  expect(wrapper.find('[data-testid="result"]').exists()).toBe(true);
});
```

**E2E Test Flakiness:**

```javascript
// Use reliable selectors
await page.click('data-testid=submit-button');

// Add intelligent waiting
await page.waitForSelector('[data-testid="results"]', {
  state: 'visible',
  timeout: 15000,
});
```

This revised guide provides comprehensive, practical testing strategies that reflect real-world SaaS application requirements.
