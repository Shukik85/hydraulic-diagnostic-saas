/**
 * useApi.ts - Typed API client composable
 * Предоставляет SSR-safe методы для работы с API
 */

import type { UseFetchOptions } from '#app'

/**
 * Интерфейс для API клиента
 */
export interface ApiClient {
  /**
   * GET запрос
   * @param url - Относительный путь API
   * @param options - Дополнительные опции fetch
   */
  get<T = any>(url: string, options?: UseFetchOptions<T>): Promise<T>
  
  /**
   * POST запрос
   * @param url - Относительный путь API
   * @param data - Данные для отправки
   * @param options - Дополнительные опции fetch
   */
  post<T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T>
  
  /**
   * PUT запрос
   * @param url - Относительный путь API
   * @param data - Данные для обновления
   * @param options - Дополнительные опции fetch
   */
  put<T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T>
  
  /**
   * PATCH запрос
   * @param url - Относительный путь API
   * @param data - Данные для частичного обновления
   * @param options - Дополнительные опции fetch
   */
  patch<T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T>
  
  /**
   * DELETE запрос
   * @param url - Относительный путь API
   * @param options - Дополнительные опции fetch
   */
  delete<T = any>(url: string, options?: UseFetchOptions<T>): Promise<T>
}

/**
 * API клиент composable
 * Использует $fetch для client-side запросов
 * Для SSR-safe запросов используйте useFetch напрямую
 * 
 * @example
 * ```typescript
 * const api = useApi()
 * 
 * // Client-side запрос
 * const data = await api.get<SystemResponse>('/systems')
 * 
 * // SSR-safe запрос (используйте useFetch)
 * const { data: systems } = await useFetch('/api/systems')
 * ```
 */
export const useApi = (): ApiClient => {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase as string
  
  return {
    get: async <T = any>(url: string, options?: UseFetchOptions<T>): Promise<T> => {
      return await $fetch<T>(url, { 
        baseURL, 
        method: 'GET',
        ...options 
      })
    },
    
    post: async <T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T> => {
      return await $fetch<T>(url, { 
        baseURL, 
        method: 'POST', 
        body: data,
        ...options 
      })
    },
    
    put: async <T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T> => {
      return await $fetch<T>(url, { 
        baseURL, 
        method: 'PUT', 
        body: data,
        ...options 
      })
    },
    
    patch: async <T = any>(url: string, data?: any, options?: UseFetchOptions<T>): Promise<T> => {
      return await $fetch<T>(url, { 
        baseURL, 
        method: 'PATCH', 
        body: data,
        ...options 
      })
    },
    
    delete: async <T = any>(url: string, options?: UseFetchOptions<T>): Promise<T> => {
      return await $fetch<T>(url, { 
        baseURL, 
        method: 'DELETE',
        ...options 
      })
    },
  }
}

/**
 * Helper для создания SSR-safe запросов с автоматическим кэшированием
 * 
 * @example
 * ```typescript
 * const { data, pending, error, refresh } = await useApiFetch<System[]>('/systems', {
 *   key: 'systems-list'
 * })
 * ```
 */
export const useApiFetch = <T = any>(
  url: string, 
  options?: UseFetchOptions<T>
) => {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase as string
  
  return useFetch<T>(url, {
    baseURL,
    getCachedData: (key) => useNuxtApp().payload.data[key as any],
    lazy: true,
    retry: 3,
    retryDelay: 1000,
    ...options
  })
}
