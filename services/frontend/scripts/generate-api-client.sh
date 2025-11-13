#!/bin/bash
# services/frontend/scripts/generate-api-client.sh
# Generate TypeScript API client from OpenAPI spec

set -e

echo "🎯 Generating TypeScript API client from OpenAPI spec..."

# Check if spec exists
if [ ! -f "../../specs/combined-api.json" ]; then
    echo "❌ OpenAPI spec not found: ../../specs/combined-api.json"
    echo "   Run: npm run generate:specs first"
    exit 1
fi

# Validate spec
echo "📋 Validating OpenAPI specification..."
if npx swagger-cli validate ../../specs/combined-api.json; then
    echo "✅ OpenAPI spec is valid"
else
    echo "❌ Invalid OpenAPI spec"
    exit 1
fi

# Clean previous generated code
if [ -d "generated/api" ]; then
    echo "🧹 Cleaning previous generated code..."
    rm -rf generated/api
fi

# Generate TypeScript client
echo "⚙️  Generating TypeScript client..."
npx openapi-typescript-codegen \
    --input ../../specs/combined-api.json \
    --output ./generated/api \
    --client axios \
    --useOptions \
    --useUnionTypes \
    --exportCore true \
    --exportServices true \
    --exportModels true \
    --exportSchemas false

if [ $? -eq 0 ]; then
    echo "✅ TypeScript client generated successfully"
    echo ""
    echo "📊 Generation stats:"
    echo "   Models: $(find generated/api/models -type f | wc -l)"
    echo "   Services: $(find generated/api/services -type f | wc -l)"
    echo "   Total files: $(find generated/api -type f | wc -l)"
    echo ""
    echo "📦 Import examples:"
    echo "   import { DiagnosisService } from '~/generated/api/services'"
    echo "   import type { DiagnosisResult } from '~/generated/api/models'"
else
    echo "❌ Failed to generate TypeScript client"
    exit 1
fi

# Generate index file for easy imports
echo "📝 Creating index exports..."
cat > generated/api/index.ts << 'EOF'
/**
 * Auto-generated API client from OpenAPI specification.
 * 
 * @packageDocumentation
 */

// Export all models
export * from './models'

// Export all services
export * from './services'

// Export core configuration
export { Configuration, type ConfigurationParameters } from './core/OpenAPI'

// Re-export commonly used types
export type {
  DiagnosisResult,
  RAGInterpretation,
  Equipment,
  ComponentHealth,
  Anomaly
} from './models'
EOF

echo "✅ Index exports created"
echo ""
echo "🎉 API client generation complete!"
echo ""
echo "Next steps:"
echo "  1. Import in your code: import { useApi } from '~/composables/useGeneratedApi'"
echo "  2. Use typed clients: const { diagnosis } = useApi()"
echo "  3. Enjoy autocomplete and type safety! 🚀"
echo ""
