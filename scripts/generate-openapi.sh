#!/bin/bash
# scripts/generate-openapi.sh
# Generate OpenAPI specs from all FastAPI services

set -e

echo "🔧 Generating OpenAPI specifications..."

# Create specs directory
mkdir -p specs

# Service definitions
declare -A SERVICES=(
    ["equipment"]="8002"
    ["diagnosis"]="8003"
    ["gnn"]="8002"
    ["rag"]="8004"
)

# Check if services are running
echo "📊 Checking service health..."
for service in "${!SERVICES[@]}"; do
    port=${SERVICES[$service]}
    if curl -f -s "http://localhost:${port}/health" > /dev/null; then
        echo "✅ ${service}-service (port ${port}) is healthy"
    else
        echo "❌ ${service}-service (port ${port}) is not responding"
        echo "   Start services with: docker-compose up -d"
        exit 1
    fi
done

# Download OpenAPI specs
echo ""
echo "📥 Downloading OpenAPI specifications..."
for service in "${!SERVICES[@]}"; do
    port=${SERVICES[$service]}
    spec_file="specs/${service}-service.json"
    
    echo "  Fetching ${service}-service..."
    curl -s "http://localhost:${port}/openapi.json" > "${spec_file}"
    
    # Validate JSON
    if jq empty "${spec_file}" 2>/dev/null; then
        echo "  ✅ ${spec_file} ($(wc -c < ${spec_file} | awk '{print int($1/1024)}')KB)"
    else
        echo "  ❌ Invalid JSON in ${spec_file}"
        exit 1
    fi
done

# Merge specs
echo ""
echo "🔗 Merging specifications..."
npx openapi-merge-cli \
    --input 'specs/*-service.json' \
    --output 'specs/combined-api.json'

if [ -f "specs/combined-api.json" ]; then
    echo "✅ Combined spec created: specs/combined-api.json"
    echo "   Size: $(wc -c < specs/combined-api.json | awk '{print int($1/1024)}')KB"
    echo "   Endpoints: $(jq '.paths | length' specs/combined-api.json)"
else
    echo "❌ Failed to create combined spec"
    exit 1
fi

# Validate combined spec
echo ""
echo "✓ Validating combined specification..."
if npx swagger-cli validate specs/combined-api.json 2>/dev/null; then
    echo "✅ Combined spec is valid"
else
    echo "⚠️  Validation warnings (may still work)"
fi

# Generate TypeScript client
echo ""
echo "🎯 Generating TypeScript client..."
cd services/frontend

if npm run generate:api; then
    echo "✅ TypeScript client generated successfully"
    echo "   Location: services/frontend/generated/api/"
    echo "   Files: $(find generated/api -type f | wc -l)"
else
    echo "❌ Failed to generate TypeScript client"
    exit 1
fi

cd ../..

echo ""
echo "🎉 OpenAPI generation complete!"
echo ""
echo "Next steps:"
echo "  1. Review generated files in services/frontend/generated/api/"
echo "  2. Import types: import { DiagnosisResult } from '~/generated/api/models'"
echo "  3. Use clients: const api = useApi(); await api.diagnosis.runDiagnosis(...)"
echo ""
