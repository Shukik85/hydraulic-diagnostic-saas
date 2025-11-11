#!/bin/bash
set -e

echo "🔄 Aggregating OpenAPI specs..."
python tools/aggregate_openapi.py

echo "📦 Generating TypeScript API clients..."
cd services/frontend

# Используем @openapitools/openapi-generator-cli
npx @openapitools/openapi-generator-cli@latest generate \
  -i ../../docs/api/openapi.yaml \
  -g typescript-axios \
  -o api/generated \
  --additional-properties=supportsES6=true,npmName=@hydraulic/api-client

echo "✅ Frontend API clients generated successfully!"
echo "📁 Location: services/frontend/api/generated/"
