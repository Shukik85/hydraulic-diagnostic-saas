#!/bin/bash
# E2E Integration Test Script for Microservices
# Проверяет backend ↔ ml_service ↔ rag_service integration

set -e

echo "======================================"
echo "E2E Microservices Integration Test"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Variables
BACKEND_URL="http://localhost:8000"
ML_SERVICE_URL="http://ml_service:8001"
RAG_SERVICE_URL="http://rag_service:8002"

echo "${YELLOW}Step 1: Health Checks${NC}"
echo "--------------------------------------"

# Backend health
echo -n "Backend health... "
if curl -f -s "${BACKEND_URL}/health/" > /dev/null; then
    echo "${GREEN}OK${NC}"
else
    echo "${RED}FAILED${NC}"
    exit 1
fi

# ML Service health (via internal network)
echo -n "ML Service health... "
if docker exec hdx-backend curl -f -s "${ML_SERVICE_URL}/health" > /dev/null; then
    echo "${GREEN}OK${NC}"
else
    echo "${RED}FAILED${NC}"
    exit 1
fi

# RAG Service health (via internal network)
echo -n "RAG Service health... "
if docker exec hdx-backend curl -f -s "${RAG_SERVICE_URL}/health" > /dev/null; then
    echo "${GREEN}OK${NC}"
else
    echo "${RED}FAILED${NC}"
    exit 1
fi

echo ""
echo "${YELLOW}Step 2: Authentication${NC}"
echo "--------------------------------------"

# Create test user if needed
echo "Creating test user..."
docker exec hdx-backend python manage.py shell << 'EOF' || true
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='testuser').exists():
    User.objects.create_superuser('testuser', 'test@example.com', 'testpass123')
    print("Test user created")
else:
    print("Test user already exists")
EOF

# Get JWT token
echo "Getting JWT token..."
TOKEN_RESPONSE=$(curl -s -X POST "${BACKEND_URL}/api/auth/login/" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }')

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access' 2>/dev/null)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    echo "${RED}Failed to get JWT token${NC}"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

echo "${GREEN}JWT token obtained${NC}"

echo ""
echo "${YELLOW}Step 3: ML Service Integration Test${NC}"
echo "--------------------------------------"

# Test ML prediction via backend gateway
echo "Testing ML prediction via backend gateway..."
ML_RESPONSE=$(curl -s -X POST "${BACKEND_URL}/api/diagnostics/anomaly/detect/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": 1,
    "sensor_data": {
      "pressure": [100.5, 101.2, 99.8, 100.1, 100.3],
      "temperature": [45.3, 45.1, 45.5, 45.2, 45.4],
      "flow": [25.0, 24.8, 25.2, 25.1, 24.9],
      "vibration": [0.5, 0.6, 0.5, 0.5, 0.6]
    }
  }')

if echo "$ML_RESPONSE" | jq -e '.prediction' > /dev/null 2>&1; then
    echo "${GREEN}ML prediction successful${NC}"
    echo "Anomaly score: $(echo $ML_RESPONSE | jq -r '.ensemble_score')"
    echo "Processing time: $(echo $ML_RESPONSE | jq -r '.total_processing_time_ms')ms"
else
    echo "${RED}ML prediction failed${NC}"
    echo "Response: $ML_RESPONSE"
    exit 1
fi

echo ""
echo "${YELLOW}Step 4: RAG Service Integration Test${NC}"
echo "--------------------------------------"

# Test RAG query via backend gateway
echo "Testing RAG query via backend gateway..."
RAG_RESPONSE=$(curl -s -X POST "${BACKEND_URL}/api/rag/query/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to fix hydraulic pressure drop?",
    "system_id": 1,
    "max_results": 3
  }')

if echo "$RAG_RESPONSE" | jq -e '.response' > /dev/null 2>&1; then
    echo "${GREEN}RAG query successful${NC}"
    echo "Response length: $(echo $RAG_RESPONSE | jq -r '.response | length') chars"
    echo "Sources found: $(echo $RAG_RESPONSE | jq -r '.sources | length')"
else
    echo "${RED}RAG query failed${NC}"
    echo "Response: $RAG_RESPONSE"
    exit 1
fi

echo ""
echo "${YELLOW}Step 5: Security Tests${NC}"
echo "--------------------------------------"

# Test direct access to ml_service (should fail)
echo -n "Testing direct ML Service access (should fail)... "
if docker exec hdx-backend curl -s -X POST "${ML_SERVICE_URL}/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{}' | grep -q "403\|forbidden" 2>/dev/null; then
    echo "${GREEN}Blocked (as expected)${NC}"
else
    echo "${RED}SECURITY ISSUE: Direct access not blocked!${NC}"
    exit 1
fi

# Test with correct API key (should work)
echo -n "Testing with correct API key... "
if docker exec hdx-backend curl -s -X POST "${ML_SERVICE_URL}/api/v1/predict" \
  -H "X-Internal-API-Key: ${ML_INTERNAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {"system_id": 1},
    "use_cache": false
  }' | jq -e '.prediction' > /dev/null 2>&1; then
    echo "${GREEN}OK${NC}"
else
    echo "${YELLOW}WARNING: Check ML_INTERNAL_API_KEY${NC}"
fi

echo ""
echo "${GREEN}======================================${NC}"
echo "${GREEN}All tests passed!${NC}"
echo "${GREEN}======================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Run unit tests: docker-compose exec backend pytest"
echo "  2. Run performance tests: cd ml_service && pytest tests/performance/"
echo "  3. Check Prometheus metrics: docker exec hdx-backend curl http://ml_service:8001/metrics"
echo ""
