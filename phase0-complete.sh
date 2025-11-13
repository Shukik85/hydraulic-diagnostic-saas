#!/bin/bash
# phase0-complete.sh
# Complete Phase 0 and unblock frontend team

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Phase 0 Completion Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1/6: Checking prerequisites...${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  jq not found (optional, but recommended)${NC}"
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Step 2: Install Python dependencies
echo -e "${BLUE}Step 2/6: Installing Python dependencies...${NC}"

services=("shared" "gnn_service" "rag_service" "equipment_service" "diagnosis_service")

for service in "${services[@]}"; do
    if [ -f "services/$service/requirements.txt" ]; then
        echo -e "  Installing ${service}..."
        cd "services/$service"
        pip install -q -r requirements.txt 2>&1 | grep -v "Requirement already satisfied" || true
        cd ../..
        echo -e "${GREEN}  ✅ ${service}${NC}"
    fi
done

echo ""

# Step 3: Start services
echo -e "${BLUE}Step 3/6: Starting Docker services...${NC}"

docker-compose up -d

echo -e "${YELLOW}  Waiting for services to be healthy (30s)...${NC}"
sleep 30

echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Step 4: Health checks
echo -e "${BLUE}Step 4/6: Health checks...${NC}"

ports=("8002" "8003" "8004")
service_names=("GNN" "Diagnosis" "RAG")

all_healthy=true
for i in "${!ports[@]}"; do
    port=${ports[$i]}
    name=${service_names[$i]}
    
    if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
        status=$(curl -s "http://localhost:${port}/health" | jq -r '.status' 2>/dev/null || echo "unknown")
        if [ "$status" = "healthy" ]; then
            echo -e "${GREEN}  ✅ ${name} Service (port ${port}): ${status}${NC}"
        else
            echo -e "${YELLOW}  ⚠️  ${name} Service (port ${port}): ${status}${NC}"
        fi
    else
        echo -e "${RED}  ❌ ${name} Service (port ${port}): not responding${NC}"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo -e "${RED}\n❌ Some services are not healthy. Check logs:${NC}"
    echo "  docker-compose logs"
    exit 1
fi

echo ""

# Step 5: Verify Swagger UI
echo -e "${BLUE}Step 5/6: Verifying Swagger UI...${NC}"

for i in "${!ports[@]}"; do
    port=${ports[$i]}
    name=${service_names[$i]}
    
    if curl -s "http://localhost:${port}/docs" | grep -q "swagger" 2>/dev/null; then
        echo -e "${GREEN}  ✅ ${name}: http://localhost:${port}/docs${NC}"
    else
        echo -e "${YELLOW}  ⚠️  ${name}: Swagger UI may not be available${NC}"
    fi
done

echo ""

# Step 6: Generate OpenAPI specs
echo -e "${BLUE}Step 6/6: Generating OpenAPI specifications...${NC}"

if [ ! -f "scripts/generate-openapi.sh" ]; then
    echo -e "${RED}❌ Generation script not found${NC}"
    exit 1
fi

chmod +x scripts/generate-openapi.sh

if ./scripts/generate-openapi.sh; then
    echo -e "${GREEN}✅ OpenAPI specs generated${NC}"
else
    echo -e "${RED}❌ Spec generation failed${NC}"
    echo "Check logs above for errors"
    exit 1
fi

echo ""

# Verification
echo -e "${BLUE}Verification...${NC}"

if [ -f "specs/combined-api.json" ]; then
    spec_size=$(wc -c < specs/combined-api.json | awk '{print int($1/1024)}')
    endpoints=$(jq '.paths | length' specs/combined-api.json 2>/dev/null || echo "unknown")
    echo -e "${GREEN}  ✅ Combined spec: ${spec_size}KB, ${endpoints} endpoints${NC}"
else
    echo -e "${RED}  ❌ Combined spec not found${NC}"
    exit 1
fi

if [ -d "services/frontend/generated/api" ]; then
    file_count=$(find services/frontend/generated/api -type f | wc -l)
    echo -e "${GREEN}  ✅ TypeScript client: ${file_count} files generated${NC}"
else
    echo -e "${RED}  ❌ TypeScript client not generated${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}────────────────────────────────────────${NC}"
echo -e "${GREEN}🎉 PHASE 0 COMPLETE!${NC}"
echo -e "${GREEN}────────────────────────────────────────${NC}"
echo ""
echo -e "${GREEN}✅ All services running${NC}"
echo -e "${GREEN}✅ OpenAPI specs generated${NC}"
echo -e "${GREEN}✅ TypeScript client ready${NC}"
echo -e "${GREEN}✅ Frontend team UNBLOCKED${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. cd services/frontend"
echo "  2. npm run dev"
echo "  3. Start Issue #17 (RAG Integration)"
echo ""
echo -e "${BLUE}Frontend can now use:${NC}"
echo "  import { useGeneratedApi } from '~/composables/useGeneratedApi'"
echo "  const api = useGeneratedApi()"
echo "  const result = await api.diagnosis.runDiagnosis({ ... })"
echo ""
echo -e "${GREEN}🚀 Ready to ship!${NC}"
echo ""
