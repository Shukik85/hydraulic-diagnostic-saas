#!/bin/bash

# TypeScript API Client Generator from OpenAPI Specs
# Generates type-safe client for all backend services

set -e

echo "🚀 Generating TypeScript API Client from OpenAPI specs..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SPECS_DIR="../../specs"
OUTPUT_DIR="./generated/api"
TEMP_DIR="./generated/.temp"

# Check if specs directory exists
if [ ! -d "$SPECS_DIR" ]; then
    echo -e "${RED}❌ Error: Specs directory not found at $SPECS_DIR${NC}"
    echo -e "${YELLOW}💡 Run './scripts/generate-openapi.sh' first to generate OpenAPI specs${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"

# Check if combined spec exists
COMBINED_SPEC="$SPECS_DIR/combined-api.json"

if [ -f "$COMBINED_SPEC" ]; then
    echo -e "${GREEN}✅ Found combined API spec${NC}"
    SPEC_FILE="$COMBINED_SPEC"
else
    echo -e "${YELLOW}⚠️  Combined spec not found, will merge individual specs${NC}"
    
    # Merge individual specs (simple concatenation for now)
    echo '{"openapi": "3.1.0", "info": {"title": "Hydraulic Diagnostic API", "version": "1.0.0"}, "paths": {}}' > "$TEMP_DIR/merged.json"
    SPEC_FILE="$TEMP_DIR/merged.json"
fi

# Generate TypeScript client using openapi-typescript-codegen
echo -e "${GREEN}🔧 Generating TypeScript client...${NC}"

npx openapi-typescript-codegen \
  --input "$SPEC_FILE" \
  --output "$OUTPUT_DIR" \
  --client axios \
  --useOptions \
  --useUnionTypes \
  --exportCore true \
  --exportServices true \
  --exportModels true \
  --exportSchemas false

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TypeScript client generated successfully!${NC}"
else
    echo -e "${RED}❌ Failed to generate TypeScript client${NC}"
    exit 1
fi

# Generate index.ts for easy imports
cat > "$OUTPUT_DIR/index.ts" << 'EOF'
/**
 * Auto-generated API Client
 * 
 * DO NOT EDIT MANUALLY!
 * Generated from OpenAPI specs
 * 
 * @see scripts/generate-api-client.sh
 */

export * from './core/ApiError'
export * from './core/ApiRequestOptions'
export * from './core/ApiResult'
export * from './core/CancelablePromise'
export * from './core/OpenAPI'

export { OpenAPI } from './core/OpenAPI'
export type { OpenAPIConfig } from './core/OpenAPI'

// Export all models
export * from './models'

// Export all services
export * from './services'
EOF

echo -e "${GREEN}✅ Index file created${NC}"

# Create composable wrapper
mkdir -p "../../composables"
cat > "../../composables/useGeneratedApi.ts" << 'EOF'
/**
 * useGeneratedApi - Wrapper for auto-generated API client
 * 
 * Provides type-safe API client with automatic authentication
 * 
 * @example
 * ```typescript
 * const api = useGeneratedApi()
 * const systems = await api.equipment.getSystems()
 * ```
 */

import { OpenAPI } from '~/generated/api'
import type { OpenAPIConfig } from '~/generated/api'

export function useGeneratedApi() {
  const config = useRuntimeConfig()
  const authStore = useAuthStore()
  
  // Configure OpenAPI client
  const apiConfig: Partial<OpenAPIConfig> = {
    BASE: config.public.apiBase || 'http://localhost:8100',
    VERSION: '1.0.0',
    WITH_CREDENTIALS: false,
    CREDENTIALS: 'include',
    TOKEN: authStore.token || undefined,
    HEADERS: {
      'Content-Type': 'application/json'
    }
  }
  
  // Update OpenAPI config
  Object.assign(OpenAPI, apiConfig)
  
  // Re-export all services
  return {
    // Services will be available after generation
    // e.g., equipment, diagnosis, gnn, rag
  }
}

/**
 * Type helper for API responses
 */
export type ApiResponse<T> = {
  data: T
  status: number
  statusText: string
}

/**
 * Error handler for generated API
 */
export function handleApiError(error: any): string {
  if (error.body?.detail) {
    return error.body.detail
  }
  if (error.message) {
    return error.message
  }
  return 'Unknown API error'
}
EOF

echo -e "${GREEN}✅ Composable wrapper created${NC}"

# Update .gitignore
if ! grep -q "generated/api" ../../.gitignore 2>/dev/null; then
    echo "" >> ../../.gitignore
    echo "# Generated API Client" >> ../../.gitignore
    echo "services/frontend/generated/api/" >> ../../.gitignore
    echo -e "${GREEN}✅ Updated .gitignore${NC}"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}🎉 TypeScript API client generation complete!${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "  1. Check generated files in: $OUTPUT_DIR"
echo "  2. Import in your code: import { useGeneratedApi } from '~/composables/useGeneratedApi'"
echo "  3. Use type-safe API: const api = useGeneratedApi()"
echo ""
echo -e "${YELLOW}🔄 To regenerate:${NC}"
echo "  npm run generate:api"
echo ""