# OpenAPI Integration Guide

## 🎯 Overview

Проект использует **OpenAPI 3.1** для автоматической генерации TypeScript клиентов и синхронизации Frontend ↔ Backend.

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│   FastAPI Services (Backend)          │
│  - Equipment Service                  │
│  - Diagnosis Service                  │
│  - GNN Service                        │
│  - RAG Service                        │
└──────────────┬─────────────────────────┘
               │
               │ Auto-generate
               ↓
┌────────────────────────────────────────┐
│   OpenAPI 3.1 Specifications          │
│  - equipment-service.json             │
│  - diagnosis-service.json             │
│  - gnn-service.json                   │
│  - rag-service.json                   │
│  → combined-api.json (merged)         │
└──────────────┬─────────────────────────┘
               │
               │ openapi-typescript-codegen
               ↓
┌────────────────────────────────────────┐
│   Generated TypeScript Client         │
│  services/frontend/generated/api/     │
│  ├── models/  (все типы)              │
│  ├── services/ (API clients)          │
│  └── core/ (configuration)            │
└────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Generate OpenAPI Specs

```bash
# Start all services
docker-compose up -d

# Generate specs
./scripts/generate-openapi.sh

# Result:
# specs/
# ├── equipment-service.json
# ├── diagnosis-service.json
# ├── gnn-service.json
# ├── rag-service.json
# └── combined-api.json
```

### 2. Generate TypeScript Client

```bash
cd services/frontend
npm run generate:api

# Result:
# generated/api/
# ├── index.ts
# ├── models/
# │   ├── DiagnosisResult.ts
# │   ├── RAGInterpretation.ts
# │   └── ...
# ├── services/
# │   ├── DiagnosisService.ts
# │   ├── RAGService.ts
# │   └── ...
# └── core/
```

### 3. Use in Code

```typescript
// Import generated types and services
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { DiagnosisResult } from '~/generated/api/models'

// In component
const { diagnosis, rag } = useGeneratedApi()

// Fully typed API calls!
const result: DiagnosisResult = await diagnosis.runDiagnosis({
  equipmentId: 'exc_001',
  diagnosisRequest: {
    timeWindow: {
      startTime: '2025-11-01T00:00:00Z',
      endTime: '2025-11-13T00:00:00Z'
    }
  }
})

// RAG interpretation
const interpretation = await rag.interpretDiagnosis({
  gnnResult: result.gnn,
  equipmentContext: {
    equipment_id: result.equipment_id,
    equipment_type: 'excavator'
  }
})
```

## 🔄 CI/CD Automation

### GitHub Actions Workflow

**File**: `.github/workflows/openapi-sync.yml`

**Triggers**:
- Push to backend service files
- Pull requests
- Manual trigger

**Process**:
1. Start services
2. Download OpenAPI specs
3. Merge specs
4. Generate TypeScript client
5. Commit if changed
6. Check for breaking changes

### Breaking Changes Detection

```bash
# Compare specs
npx oasdiff breaking \
  specs/combined-api.previous.json \
  specs/combined-api.json

# If breaking changes:
# → Comment on PR
# → Block merge (optional)
# → Require manual review
```

## 📋 Development Workflow

### Backend Developer

```python
# 1. Update FastAPI endpoint
@app.post("/diagnosis")
async def run_diagnosis(request: DiagnosisRequest) -> DiagnosisResult:
    """Run diagnosis with full docstring."""
    pass

# 2. Add examples
class DiagnosisRequest(BaseModel):
    equipment_id: str = Field(..., example="exc_001")
    time_window: TimeWindow
    
    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "exc_001",
                "time_window": {
                    "start_time": "2025-11-01T00:00:00Z",
                    "end_time": "2025-11-13T00:00:00Z"
                }
            }
        }

# 3. Commit code
git add services/diagnosis_service/
git commit -m "feat: add new diagnosis endpoint"
git push

# 4. CI automatically:
#    - Generates new OpenAPI spec
#    - Updates TypeScript client
#    - Commits to repo
#    - Frontend team gets update!
```

### Frontend Developer

```bash
# 1. Pull latest code
git pull origin feature/your-branch

# 2. Install dependencies (includes generated client)
npm install

# 3. Start dev server (auto-generates client)
npm run dev

# 4. Use typed API
# TypeScript autocomplete just works! ✨
```

## 🧪 Testing

### Mock Server from OpenAPI

```typescript
// tests/setup/mock-server.ts
import { createMockServer } from '@stoplight/prism-http'
import openApiSpec from '~/generated/openapi.json'

export const mockServer = createMockServer({
  spec: openApiSpec,
  cors: true,
  port: 4010
})

// Automatically returns valid responses based on examples!
```

### Schema Validation in Tests

```typescript
// tests/unit/api-client.spec.ts
import { validateAgainstSchema } from '@openapi-contrib/openapi-schema-validator'
import openApiSpec from '~/generated/openapi.json'

test('diagnosis response matches schema', async () => {
  const response = await api.diagnosis.runDiagnosis({ ... })
  
  const validation = validateAgainstSchema(
    response,
    openApiSpec.components.schemas.DiagnosisResult
  )
  
  expect(validation.valid).toBe(true)
})
```

## 🔧 Configuration

### Backend: Enable OpenAPI

```python
# services/*/main.py
from fastapi import FastAPI
from openapi_config import custom_openapi, add_openapi_examples

app = FastAPI(
    title="Service Name",
    version="1.0.0",
    openapi_version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)
add_openapi_examples(app)
```

### Frontend: Configure Client

```typescript
// services/frontend/nuxt.config.ts
export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      apiBase: 'https://api.hydraulic-diagnostics.com',
      disableApiValidation: false,  // Enable validation in dev
      strictValidation: true  // Throw errors on validation failures
    }
  }
})
```

## 📊 Benefits

### Type Safety
```typescript
// ❌ Before (manual):
const result = await $fetch('/api/diagnosis', {
  body: {
    equpment_id: 'exc_001'  // Typo! No error until runtime
  }
})

// ✅ After (generated):
const result = await api.diagnosis.runDiagnosis({
  equipmentId: 'exc_001'  // TypeScript error if typo!
  // ^^^^^^^^^^^^^ autocomplete works!
})
```

### Documentation
```typescript
// Hover over method:
api.diagnosis.runDiagnosis(
  // Shows full docstring from backend!
  // Including examples, parameters, response types
)
```

### Testing
```typescript
// Auto-generated mocks from examples:
const mockData = openApiSpec.components.examples.DiagnosisResult.value
// Use in tests - always valid!
```

## 🐛 Troubleshooting

### Issue: Spec generation fails

```bash
# Check services are running
docker-compose ps

# Check health endpoints
curl http://localhost:8002/health
curl http://localhost:8003/health

# View service logs
docker-compose logs equipment-service
```

### Issue: Client generation fails

```bash
# Validate spec first
npx swagger-cli validate specs/combined-api.json

# Clean and regenerate
rm -rf services/frontend/generated/api
npm run generate:api
```

### Issue: Type errors after generation

```bash
# Ensure spec is valid
npm run validate:api

# Check TypeScript config
npx nuxi typecheck

# Rebuild
npm run build
```

## 🔗 Resources

- [OpenAPI 3.1 Specification](https://spec.openapis.org/oas/v3.1.0)
- [FastAPI OpenAPI](https://fastapi.tiangolo.com/advanced/extending-openapi/)
- [openapi-typescript-codegen](https://github.com/ferdikoomen/openapi-typescript-codegen)
- [Swagger UI](https://swagger.io/tools/swagger-ui/)

## ✅ Checklist

### Backend
- [ ] OpenAPI docstrings added to all endpoints
- [ ] Request/response models defined
- [ ] Examples added to schemas
- [ ] Security schemes configured
- [ ] Tags organized
- [ ] Swagger UI accessible at /docs

### Frontend
- [ ] openapi-typescript-codegen installed
- [ ] Generation script configured
- [ ] Generated code in .gitignore
- [ ] useGeneratedApi composable created
- [ ] All manual API code removed
- [ ] Types imported from generated/

### CI/CD
- [ ] openapi-sync.yml workflow added
- [ ] Breaking change detection enabled
- [ ] Auto-commit configured
- [ ] PR comments enabled

## 🎉 Success Metrics

- ✅ 100% API coverage в OpenAPI specs
- ✅ 0 manual type definitions
- ✅ < 1 minute sync time
- ✅ 0 type mismatches in production
- ✅ 80% reduction in API integration bugs
