#!/bin/bash

echo "��� Testing Phase 0 Services..."
echo ""

# Test Equipment Service
echo "1️⃣ Equipment Service (8002):"
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "   ✅ Healthy"
else
    echo "   ❌ Not responding"
fi

# Test Diagnosis Service
echo "2️⃣ Diagnosis Service (8003):"
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "   ✅ Healthy"
else
    echo "   ❌ Not responding"
fi

# Check OpenAPI specs
echo "3️⃣ OpenAPI Specs:"
if [ -f "equipment-service-spec.json" ]; then
    echo "   ✅ Equipment spec exists"
else
    echo "   ❌ Equipment spec missing"
fi

if [ -f "diagnosis-service-spec.json" ]; then
    echo "   ✅ Diagnosis spec exists"
else
    echo "   ❌ Diagnosis spec missing"
fi

echo ""
echo "��� Next: Generate TypeScript client"
echo "   cd services/frontend"
echo "   npm run generate:api"
