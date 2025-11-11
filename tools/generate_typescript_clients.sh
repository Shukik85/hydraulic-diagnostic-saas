#!/bin/bash
# Generate TypeScript clients from OpenAPI specs
# Usage: ./generate_typescript_clients.sh

set -e

echo "============================================"
echo "  Generating TypeScript API Clients"
echo "============================================"
echo ""

# Check if openapi-generator-cli is installed
if ! command -v openapi-generator-cli &> /dev/null; then
    echo "Installing openapi-generator-cli..."
    npm install -g @openapitools/openapi-generator-cli
fi

# Output directory
OUTPUT_DIR="services/frontend/composables/api/generated"
mkdir -p "$OUTPUT_DIR"

# Generate clients for each service
generate_client() {
    local service=$1
    local spec_file="docs/openapi/${service}.json"

    if [ ! -f "$spec_file" ]; then
        echo "⚠️  Spec not found: $spec_file"
        return 1
    fi

    echo "Generating client for $service..."

    openapi-generator-cli generate \
        -i "$spec_file" \
        -g typescript-fetch \
        -o "${OUTPUT_DIR}/${service}" \
        --additional-properties=npmName="@hydraulic/${service}-client",supportsES6=true,typescriptThreePlus=true

    echo "✅ $service client generated"
}

# Generate clients
generate_client "backend_fastapi"
generate_client "gnn_service"
generate_client "rag_service"

echo ""
echo "============================================"
echo "  ✅ TypeScript Client Generation Complete"
echo "============================================"
echo ""
echo "Clients saved to:"
echo "  services/frontend/composables/api/generated/"
echo ""
echo "Usage in Nuxt:"
echo "  import { DefaultApi } from '~/composables/api/generated/backend_fastapi'"
