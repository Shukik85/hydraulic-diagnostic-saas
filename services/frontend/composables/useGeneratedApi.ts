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
import { useAuthStore } from '~/stores/auth.store'

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
  const config = useRuntimeConfig()
  const authStore = useAuthStore()
  
  // Base configuration
  const configParams: ConfigurationParameters = {
    basePath: config.public.apiBase as string,
    
    // Auth token injection
    accessToken: () => {
      try {
        return authStore?.token || ''
      } catch (error) {
        console.warn('Auth store not available:', error)
        return ''
      }
    },
    
    // Custom headers
    headers: {
      'X-Device-Fingerprint': getDeviceFingerprint(),
      'X-Client-Version': '1.0.0',
      'X-Tenant-ID': authStore?.tenantId || 'default'
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
 * Генерирует уникальный отпечаток устройства для безопасности.
 * 
 * @returns Base64-encoded device fingerprint
 */
function getDeviceFingerprint(): string {
  // Server-side rendering
  if (process.server) {
    return 'server-render'
  }
  
  // Client-side
  try {
    const fingerprint = {
      userAgent: navigator.userAgent || 'unknown',
      language: navigator.language || 'unknown',
      platform: navigator.platform || 'unknown',
      screenResolution: `${screen.width || 0}x${screen.height || 0}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
      colorDepth: screen.colorDepth || 24,
      hardwareConcurrency: navigator.hardwareConcurrency || 4
    }
    
    return btoa(JSON.stringify(fingerprint))
  } catch (error) {
    console.warn('Failed to generate device fingerprint:', error)
    return 'fingerprint-error'
  }
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