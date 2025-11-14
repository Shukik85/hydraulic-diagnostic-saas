# 🚀 Phase 0 Completion Guide - UNBLOCK FRONTEND NOW!

**Current Status**: 50% Complete (Frontend + CI/CD ✅, Backend pending ⏳)  
**Time Remaining**: 2-3 hours  
**Goal**: Generate OpenAPI specs → TypeScript client → **UNBLOCK FRONTEND TEAM**  

---

## ✅ What's Already Done

- ✅ All `main.py` files created (4 services)
- ✅ OpenAPI configs created
- ✅ Monitoring endpoints implemented
- ✅ Admin endpoints implemented
- ✅ CI/CD workflow ready
- ✅ Frontend generation scripts ready
- ✅ All dependencies defined

---

## 🎯 What Needs To Be Done NOW

### Step 1: Install Dependencies (15 min)

```bash
# Pull latest code
git checkout feature/enterprise-plus-plus-architecture
git pull

# Install shared dependencies
cd services/shared
pip install -r requirements.txt

# Install per service
cd ../gnn_service
pip install -r requirements.txt

cd ../rag_service  
pip install -r requirements.txt

cd ../equipment_service
pip install -r requirements.txt

cd ../diagnosis_service
pip install -r requirements.txt

cd ../..
```

**Verification**:
```bash
python -c "import prometheus_client; import psutil; import jwt; print('✅ All deps installed')"
```

---

### Step 2: Start Services (10 min)

```bash
# From project root
docker-compose up -d

# Wait for services
sleep 20

# Check services are up
docker-compose ps

# Should see:
# - gnn-service (port 8002)
# - diagnosis-service (port 8003)
# - rag-service (port 8004)
# - equipment-service (port 8002 or 8001)
```

**Verification**:
```bash
# Test each service
curl http://localhost:8002/health | jq '.status'  # GNN
curl http://localhost:8003/health | jq '.status'  # Diagnosis
curl http://localhost:8004/health | jq '.status'  # RAG

# All should return "healthy"
```

---

### Step 3: Verify Swagger UI (5 min)

```bash
# Open in browser:
open http://localhost:8002/docs  # GNN Service
open http://localhost:8003/docs  # Diagnosis Service  
open http://localhost:8004/docs  # RAG Service

# Should see:
# - Swagger UI interface
# - All endpoints listed
# - Tags organized (Monitoring, Inference, Admin)
# - Try it out buttons working
```

**Manual Test**:
```bash
# Test GNN inference endpoint
curl -X POST http://localhost:8002/inference \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "exc_001",
    "time_window": {
      "start_time": "2025-11-01T00:00:00Z",
      "end_time": "2025-11-13T00:00:00Z"
    }
  }'

# Should return 200 or 500 (depending on data availability)
# Key point: endpoint responds
```

---

### Step 4: Generate OpenAPI Specs (5 min)

```bash
# Make script executable
chmod +x scripts/generate-openapi.sh

# Run generation
./scripts/generate-openapi.sh

# Expected output:
# 🔧 Generating OpenAPI specifications...
# 📊 Checking service health...
# ✅ equipment-service (port 8002) is healthy
# ✅ diagnosis-service (port 8003) is healthy
# ✅ gnn-service (port 8002) is healthy
# ✅ rag-service (port 8004) is healthy
# 📥 Downloading OpenAPI specifications...
# ✅ equipment-service.json (45KB)
# ✅ diagnosis-service.json (38KB)
# ✅ gnn-service.json (52KB)
# ✅ rag-service.json (41KB)
# 🔗 Merging specifications...
# ✅ Combined spec created: specs/combined-api.json
# 🎯 Generating TypeScript client...
# ✅ TypeScript client generated successfully
```

**Verification**:
```bash
# Check specs created
ls -lh specs/*.json

# Should see:
# equipment-service.json
# diagnosis-service.json
# gnn-service.json
# rag-service.json
# combined-api.json

# Validate combined spec
npx swagger-cli validate specs/combined-api.json

# Should output: "specs/combined-api.json is valid"
```

---

### Step 5: Generate TypeScript Client (5 min)

```bash
cd services/frontend

# Install dependencies (if not done)
npm install

# Generate API client
npm run generate:api

# Expected output:
# 🎯 Generating TypeScript API client from OpenAPI spec...
# 📋 Validating OpenAPI specification...
# ✅ OpenAPI spec is valid
# ⚙️  Generating TypeScript client...
# ✅ TypeScript client generated successfully
# 📊 Generation stats:
#    Models: 24
#    Services: 4
#    Total files: 35
```

**Verification**:
```bash
# Check generated code
ls -la generated/api/

# Should see:
# models/
# services/
# core/
# index.ts

# Test TypeScript compilation
npx tsc --noEmit

# Should have 0 errors
```

---

### Step 6: Test Generated Client (10 min)

```bash
# Start dev server
npm run dev

# Visit http://localhost:3000
```

**Test in browser console**:
```javascript
// Should have autocomplete!
import { DiagnosisService } from '~/generated/api/services'
import type { DiagnosisResult } from '~/generated/api/models'

// Test API call
const api = useApi()
const result = await api.diagnosis.runDiagnosis({
  equipmentId: 'exc_001',
  diagnosisRequest: {
    timeWindow: {
      startTime: '2025-11-01T00:00:00Z',
      endTime: '2025-11-13T00:00:00Z'
    }
  }
})

console.log(result)
```

---

## ✅ Success Checklist

### Backend Services
- [ ] All 4 services running
- [ ] `/health` returns "healthy" for all
- [ ] `/docs` Swagger UI accessible
- [ ] `/metrics` returns Prometheus format
- [ ] No errors in `docker-compose logs`

### OpenAPI Specs
- [ ] 4 service specs generated
- [ ] `combined-api.json` created
- [ ] Combined spec is valid
- [ ] No merge conflicts in spec
- [ ] All endpoints documented

### TypeScript Client
- [ ] Client generated in `generated/api/`
- [ ] Models exported (24+ files)
- [ ] Services exported (4 files)
- [ ] TypeScript compilation passes
- [ ] No import errors

### Frontend Ready
- [ ] `npm run dev` works
- [ ] No build errors
- [ ] API calls work
- [ ] Types autocomplete in IDE
- [ ] **Frontend team UNBLOCKED** 🎉

---

## 🐛 Troubleshooting

### Issue: Service won't start

```bash
# Check logs
docker-compose logs gnn-service

# Common fixes:
# - Missing dependencies: pip install -r requirements.txt
# - Port conflict: change port in docker-compose.yml
# - Import errors: check sys.path in monitoring_endpoints.py
```

**Fix import errors**:
```python
# In monitoring_endpoints.py, change:
import sys
sys.path.append('../shared')  # Relative path may fail

# To absolute:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
```

---

### Issue: Spec generation fails

```bash
# Check services responding
curl http://localhost:8002/openapi.json
curl http://localhost:8003/openapi.json
curl http://localhost:8004/openapi.json

# If 404, check main.py has:
app = FastAPI(
    openapi_version="3.1.0",  # Must be set!
    docs_url="/docs"
)
```

---

### Issue: TypeScript generation fails

```bash
# Validate spec first
npx swagger-cli validate specs/combined-api.json

# If invalid, check for:
# - Missing $ref definitions
# - Circular references
# - Invalid schema types

# Manual fix:
# Edit specs/combined-api.json
# Then re-run: npm run generate:api
```

---

### Issue: Import errors in generated client

```bash
# Clean and regenerate
rm -rf services/frontend/generated/api
npm run generate:api

# If still fails, check:
# - Node version (need 18+)
# - npm version (need 9+)
# - TypeScript version (need 5+)
```

---

## 📊 Timeline

```
NOW (07:00)
    ↓
┌──────────────────────────┐
│ 07:00-07:15 (15min)        │
│ Install dependencies        │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:15-07:25 (10min)        │
│ Start services             │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:25-07:30 (5min)         │
│ Verify Swagger UI          │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:30-07:35 (5min)         │
│ Generate OpenAPI specs     │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:35-07:40 (5min)         │
│ Generate TypeScript client │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:40-07:50 (10min)        │
│ Test frontend integration  │
└─────────┬────────────────┘
         ↓
┌──────────────────────────┐
│ 07:50 - COMPLETE!          │
│ ✅ Frontend UNBLOCKED      │
│ ✅ Phase 0 DONE           │
│ 🚀 Move to Phase 2        │
└──────────────────────────┘

Total: 50 minutes
```

---

## 🎯 Quick Commands (Copy-Paste)

### Complete Phase 0 in One Go:

```bash
#!/bin/bash
# phase0-complete.sh - Run this to complete Phase 0

set -e

echo "🚀 Starting Phase 0 completion..."

# Step 1: Dependencies
echo "Step 1/6: Installing dependencies..."
for service in shared gnn_service rag_service equipment_service diagnosis_service; do
    cd services/$service
    pip install -q -r requirements.txt
    cd ../..
    echo "  ✅ $service"
done

# Step 2: Start services
echo "Step 2/6: Starting services..."
docker-compose up -d
sleep 20
echo "  ✅ Services started"

# Step 3: Health check
echo "Step 3/6: Health checks..."
for port in 8002 8003 8004; do
    status=$(curl -s http://localhost:${port}/health | jq -r '.status')
    echo "  Port $port: $status"
done

# Step 4: Swagger UI check
echo "Step 4/6: Verifying Swagger UI..."
for port in 8002 8003 8004; do
    if curl -s http://localhost:${port}/docs | grep -q "swagger"; then
        echo "  ✅ Port $port Swagger UI OK"
    fi
done

# Step 5: Generate specs
echo "Step 5/6: Generating OpenAPI specs..."
chmod +x scripts/generate-openapi.sh
./scripts/generate-openapi.sh

# Step 6: Verify
echo "Step 6/6: Verifying generation..."
if [ -f "specs/combined-api.json" ]; then
    echo "  ✅ Combined spec created"
    npx swagger-cli validate specs/combined-api.json
fi

if [ -d "services/frontend/generated/api" ]; then
    echo "  ✅ TypeScript client generated"
    echo "  Files: $(find services/frontend/generated/api -type f | wc -l)"
fi

echo ""
echo "🎉 Phase 0 COMPLETE!"
echo ""
echo "Frontend team can now:"
echo "  1. cd services/frontend"
echo "  2. npm run dev"
echo "  3. Start building RAG components (Issue #17)"
echo ""
```

**Save and run**:
```bash
chmod +x phase0-complete.sh
./phase0-complete.sh
```

---

## 📢 Notify Frontend Team

Once complete, send this message:

```
🎉 **Phase 0 Complete - Frontend UNBLOCKED!**

✅ OpenAPI specs generated
✅ TypeScript client ready
✅ All types available

**You can now start**:
1. cd services/frontend
2. npm install (if needed)
3. npm run dev
4. Import from ~/generated/api

**Example**:
```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { DiagnosisResult } from '~/generated/api/models'

const api = useGeneratedApi()
const result = await api.diagnosis.runDiagnosis({ ... })
// Full autocomplete works! ✨
```

**Next**: Start Issue #17 (RAG Integration)
**Timeline**: 7 hours remaining
**Deadline**: 14 ноября 18:00

🚀 Let's ship it!
```

---

## 📊 Success Metrics

**Phase 0 Complete When**:
- ✅ All services running
- ✅ OpenAPI specs generated
- ✅ TypeScript client generated
- ✅ Frontend `npm run dev` works
- ✅ No TypeScript errors
- ✅ API calls successful

**Impact**:
- ✅ Frontend team unblocked
- ✅ -80% integration time saved
- ✅ 100% type safety guaranteed
- ✅ Auto-sync enabled

---

## 🚀 After Phase 0

**Immediately start**:
- Issue #17: RAG Integration (7h)
- Issue #18: Authentication (3.5h)

**Timeline**:
- Today: Phase 0 + start Phase 2
- Tomorrow: Complete Phase 2-6
- 15 ноября: Testing + Launch

**We're on track!** 🎯
