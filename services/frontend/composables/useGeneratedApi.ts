/**
 * Wrapper для auto-generated OpenAPI client.
 * Предоставляет typed API clients для всех backend сервисов с JWT интеграцией.
 *
 * @example
 * ```typescript
 * const { diagnosis, equipment, gnn, rag } = useGeneratedApi()
 * const systems = await equipment.getSystems()
 * ```
 */

import { OpenAPI } from '~/generated/api/core/OpenAPI'
import { useRuntimeConfig } from 'nuxt/app'

// Import generated services (когда будет готов OpenAPI codegen)
// import {
//   DiagnosisService,
//   EquipmentService,
//   GNNService,
//   RAGService
// } from '~/generated/api/services'

/**
 * Создает configured API clients для всех сервисов с автоматической JWT интеграцией.
 */
export const useGeneratedApi = () => {
  const config = useRuntimeConfig()
  const authStore = useAuthStore()

  // ✅ FIX: Configure OpenAPI with proper base URL
  OpenAPI.BASE = config.public.apiBase || 'http://localhost:8000/api/v1'

  // ✅ FIX: Set JWT token from auth store
  if (authStore.authToken) {
    OpenAPI.TOKEN = authStore.authToken
  }

  // Watch for auth changes and update token
  watch(
    () => authStore.authToken,
    (newToken) => {
      OpenAPI.TOKEN = newToken || undefined
    }
  )

  // ⚠️ TODO: Uncomment when OpenAPI codegen is ready
  // return {
  //   diagnosis: new DiagnosisService(),
  //   equipment: new EquipmentService(),
  //   gnn: new GNNService(),
  //   rag: new RAGService()
  // }

  // For now, return a placeholder that warns about missing services
  if (process.dev) {
    console.warn(
      '[useGeneratedApi] OpenAPI client services not yet generated. ' +
      'Run: npm run generate:api'
    )
  }

  return {
    diagnosis: null,
    equipment: null,
    gnn: null,
    rag: null
  }
}

/**
 * Типобезопасный алиас для useGeneratedApi
 * @deprecated Используйте useGeneratedApi() напрямую
 */
export const useApi = useGeneratedApi
