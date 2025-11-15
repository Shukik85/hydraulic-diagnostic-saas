// services/frontend/composables/useGeneratedApi.ts
/**
 * Wrapper для auto-generated OpenAPI client.
 * Предоставляет typed API clients для всех backend сервисов.
 */
import {
  DiagnosisService,
  EquipmentService,
  GNNService,
  RAGService,
  Configuration
} from '~/generated/api/services'

import type { ConfigurationParameters } from '~/generated/api/core/OpenAPI'
import { useRuntimeConfig } from 'nuxt/app'

/**
 * Создает configured API clients для всех сервисов.
 *
 * @example
 * ```typescript
 * const { diagnosis, rag } = useGeneratedApi()
 *
 * // Fully typed!
 * const result = await diagnosis.runDiagnosis({
 *   equipmentId: 'exc_001',
 *   diagnosisRequest: {
 *     timeWindow: {
 *       startTime: '2025-11-01T00:00:00Z',
 *       endTime: '2025-11-13T00:00:00Z'
 *     }
 *   }
 * })
 *
 * // RAG interpretation
 * const interpretation = await rag.interpretDiagnosis({
 *   gnnResult: result,
 *   equipmentContext: { ... }
 * })
 * ```
 */
export const useGeneratedApi = () => {
  const authStore = useAuthStore()
  const config = useRuntimeConfig()

  // Base configuration
  const configParams: ConfigurationParameters = {
    basePath: config.public.apiBase,

    // Auth token injection
    accessToken: () => authStore.token,

    // Custom headers
    headers: {
      'X-Device-Fingerprint': getDeviceFingerprint(),
      'X-Client-Version': '1.0.0',
      'X-Tenant-ID': authStore.tenantId
    },

    // Credentials
    credentials: 'include'
  }

  const apiConfig = new Configuration(configParams)

  return {
    diagnosis: new DiagnosisService(apiConfig),
    equipment: new EquipmentService(apiConfig),
    gnn: new GNNService(apiConfig),
    rag: new RAGService(apiConfig)
  }
}

/**
 * Helper для device fingerprinting.
 */
function getDeviceFingerprint(): string {
  if (process.server) return 'server'

  const fingerprint = {
    userAgent: navigator.userAgent,
    language: navigator.language,
    platform: navigator.platform,
    screenResolution: `${screen.width}x${screen.height}`,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    colorDepth: screen.colorDepth,
    hardwareConcurrency: navigator.hardwareConcurrency
  }

  return btoa(JSON.stringify(fingerprint))
}

/**
 * Type-safe API hook для компонентов.
 *
 * @example
 * ```vue
 * <script setup>
 * const api = useApi()
 * const systems = await api.equipment.listSystems()
 * </script>
 * ```
 */
export const useApi = useGeneratedApi  // Alias для удобства
