#!/bin/bash
# Generate OpenAPI specs from running services
# Usage: ./generate_openapi.sh

set -e

echo "============================================"
echo "  Generating OpenAPI Specifications"
echo "============================================"
echo ""

# Check if services are running
check_service() {
    local name=$1
    local url=$2

    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "✅ $name is running"
        return 0
    else
        echo "❌ $name is not running at $url"
        return 1
    fi
}

# Check all services
echo "Checking services..."
check_service "Backend FastAPI" "http://localhost:8100/health/" || {
    echo "⚠️  Start services with: docker-compose up -d"
    exit 1
}

check_service "GNN Service" "http://localhost:8001/gnn/health" || true
check_service "RAG Service" "http://localhost:8002/health" || true

echo ""
echo "Fetching OpenAPI specs..."

# Create output directory
mkdir -p docs/openapi

# Fetch specs
curl -s http://localhost:8100/openapi.json > docs/openapi/backend_fastapi.json
echo "✅ Backend FastAPI spec saved"

curl -s http://localhost:8001/openapi.json > docs/openapi/gnn_service.json 2>/dev/null || echo "⚠️  GNN Service spec not available"

curl -s http://localhost:8002/openapi.json > docs/openapi/rag_service.json 2>/dev/null || echo "⚠️  RAG Service spec not available"

echo ""
echo "Aggregating specs..."
python tools/aggregate_openapi.py

echo ""
echo "============================================"
echo "  ✅ OpenAPI Generation Complete"
echo "============================================"
echo ""
echo "Specs saved to docs/openapi/"
echo "  - aggregated.yaml (all services)"
echo "  - backend_fastapi.json"
echo "  - gnn_service.json"
echo "  - rag_service.json"
