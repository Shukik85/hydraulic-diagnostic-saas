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

import { OpenAPI } from '~/generated/api/core/OpenAPI'
import { useRuntimeConfig } from 'nuxt/app'

/**
 * Создает configured API clients для всех сервисов.
 */
// export const useGeneratedApi = () => {
//   const config = useRuntimeConfig()

//   // OpenAPI config доступен напрямую (новый openapi-typescript)
//   OpenAPI.BASE = config.public.apiBase
//   OpenAPI.TOKEN = undefined // TODO: добавить интеграцию с authStore

//   return {
//     diagnosis: new DiagnosisService(),
//     equipment: new EquipmentService(),
//     gnn: new GNNService(),
//     rag: new RAGService()
//   }
// }

/**
 * Временный API wrapper для компатибильности с существующим кодом.
 * TODO: Мигрировать на полноценные generated services.
 */
export const useGeneratedApi = () => {
  const config = useRuntimeConfig()
  const baseURL = config.public?.apiBase || 'http://localhost:8000'

  /**
   * Universal request wrapper с поддержкой query params
   */
  async function request<T = any>(
    url: string,
    options?: RequestInit & { params?: Record<string, any> }
  ): Promise<T> {
    // Build full URL с query params
    let fullUrl = `${baseURL}${url}`
    
    if (options?.params) {
      const queryString = new URLSearchParams(
        Object.entries(options.params)
          .filter(([_, value]) => value !== undefined && value !== null)
          .map(([key, value]) => [key, String(value)])
      ).toString()
      
      if (queryString) {
        fullUrl += `?${queryString}`
      }
    }

    try {
      const response = await fetch(fullUrl, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers
        }
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`)
      }

      const contentType = response.headers.get('content-type')
      if (contentType?.includes('application/json')) {
        return await response.json()
      }
      
      return await response.text() as T
    } catch (error) {
      console.error('API request failed:', { url: fullUrl, error })
      throw error
    }
  }

  return {
    // Основной request wrapper
    request,
    
    // Auth methods (заглушки до интеграции)
    login: async () => undefined,
    register: async () => undefined,
    logout: async () => undefined,
    isAuthenticated: async () => true,
    getCurrentUser: async () => null,
    updateUser: async () => undefined,
    
    // Nested api object для systems.store.ts
    api: { request },
    
    // Service stubs (для компатибильности)
    diagnosis: {},
    equipment: {},
    gnn: {},
    rag: {}
  }
}
