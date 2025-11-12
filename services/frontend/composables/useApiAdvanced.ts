/**
 * useApiAdvanced.ts — Production-ready API client для Hydraulic Diagnostic Platform
 * 
 * Features:
 * - Automatic retry with exponential backoff
 * - Token refresh queue
 * - Request deduplication
 * - Response caching for GET requests
 * - Batch requests
 * - Timeout handling
 * - HTTP status-specific handlers
 */
import { ref, computed } from 'vue'
import { useStorage } from '@vueuse/core'

interface ApiError {
  code: number
  message: string
  timestamp: string
  request_id: string
  details?: Record<string, any>
}

interface RetryConfig {
  maxRetries: number
  retryDelay: number
  retryableStatusCodes: number[]
  backoffMultiplier: number
}

interface ApiRequestOptions {
  signal?: AbortSignal
  headers?: Record<string, string>
  timeout?: number
  retry?: Partial<RetryConfig>
  cache?: boolean
  cacheKey?: string
  onUploadProgress?: (progress: number) => void
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504],
  backoffMultiplier: 2
}

export function useApiAdvanced() {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8100/api'
  const token = useStorage<string | null>('auth_token', null)
  const refreshToken = useStorage<string | null>('refresh_token', null)
  
  // Request cache для GET запросов
  const requestCache = new Map<string, { data: any; timestamp: number }>()
  const CACHE_TTL = 5 * 60 * 1000 // 5 минут
  
  // Tracking активных запросов для дедупликации
  const pendingRequests = new Map<string, Promise<any>>()
  
  const isRefreshing = ref(false)
  const refreshSubscribers: Array<(token: string) => void> = []

  /**
   * HTTP Status Code Handlers
   */
  const statusHandlers: Record<number, (error: ApiError, url: string) => Promise<boolean>> = {
    401: async (error, url) => {
      // Unauthorized - refresh token
      if (!isRefreshing.value) {
        return await handleTokenRefresh()
      }
      return new Promise(resolve => {
        refreshSubscribers.push((newToken: string) => {
          token.value = newToken
          resolve(true)
        })
      })
    },
    403: async (error) => {
      // Forbidden - redirect to permission denied
      console.error('Access denied:', error.message)
      if (process.client) {
        await navigateTo('/error/forbidden')
      }
      return false
    },
    404: async (error, url) => {
      console.warn(`Resource not found: ${url}`)
      return false
    },
    429: async (error) => {
      // Rate limiting - exponential backoff
      const retryAfter = parseInt(error.details?.retry_after || '60', 10)
      console.warn(`Rate limited. Retry after ${retryAfter}s`)
      await sleep(retryAfter * 1000)
      return true
    },
    500: async (error) => {
      console.error('Server error:', error)
      return true // Retryable
    },
    502: async () => true,
    503: async () => true,
    504: async () => true
  }

  /**
   * Token refresh logic with queue
   */
  async function handleTokenRefresh(): Promise<boolean> {
    if (!refreshToken.value) {
      await navigateTo('/auth/login')
      return false
    }

    isRefreshing.value = true

    try {
      const response = await fetch(`${API_URL}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken.value })
      })

      if (!response.ok) {
        throw new Error('Token refresh failed')
      }

      const data = await response.json()
      token.value = data.access_token
      
      // Notify all waiting requests
      refreshSubscribers.forEach(callback => callback(data.access_token))
      refreshSubscribers.length = 0
      
      return true
    } catch (error) {
      console.error('Token refresh error:', error)
      token.value = null
      refreshToken.value = null
      await navigateTo('/auth/login')
      return false
    } finally {
      isRefreshing.value = false
    }
  }

  /**
   * Main request function with retry logic
   */
  async function request<T>(
    endpoint: string,
    options: ApiRequestOptions & {
      method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
      body?: any
    } = {}
  ): Promise<T> {
    const url = endpoint.startsWith('http') ? endpoint : `${API_URL}${endpoint}`
    const method = options.method || 'GET'
    const retryConfig = { ...DEFAULT_RETRY_CONFIG, ...options.retry }
    
    // Cache key для дедупликации и кэширования
    const cacheKey = options.cacheKey || `${method}:${url}:${JSON.stringify(options.body || {})}`
    
    // Проверка кэша для GET запросов
    if (method === 'GET' && options.cache !== false) {
      const cached = requestCache.get(cacheKey)
      if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.data as T
      }
    }
    
    // Дедупликация одинаковых запросов
    if (pendingRequests.has(cacheKey)) {
      return pendingRequests.get(cacheKey) as Promise<T>
    }
    
    const executeRequest = async (attemptNumber = 0): Promise<T> => {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...options.headers
      }
      
      if (token.value) {
        headers['Authorization'] = `Bearer ${token.value}`
      }
      
      const controller = new AbortController()
      const timeoutId = options.timeout
        ? setTimeout(() => controller.abort(), options.timeout)
        : null
      
      try {
        const response = await fetch(url, {
          method,
          headers,
          body: options.body ? JSON.stringify(options.body) : undefined,
          signal: options.signal || controller.signal
        })
        
        if (timeoutId) clearTimeout(timeoutId)
        
        // Parse response
        const contentType = response.headers.get('content-type')
        const isJson = contentType?.includes('application/json')
        const data = isJson ? await response.json() : await response.text()
        
        if (response.ok) {
          // Кэшируем успешные GET запросы
          if (method === 'GET' && options.cache !== false) {
            requestCache.set(cacheKey, { data, timestamp: Date.now() })
          }
          
          pendingRequests.delete(cacheKey)
          return data as T
        }
        
        // Handle error response
        const error: ApiError = {
          code: response.status,
          message: data.message || data.error || response.statusText,
          timestamp: new Date().toISOString(),
          request_id: response.headers.get('x-request-id') || '',
          details: isJson ? data : {}
        }
        
        // Check if status is retryable
        const handler = statusHandlers[response.status]
        const shouldRetry = handler ? await handler(error, url) : false
        
        if (
          shouldRetry &&
          attemptNumber < retryConfig.maxRetries &&
          retryConfig.retryableStatusCodes.includes(response.status)
        ) {
          const delay = retryConfig.retryDelay * Math.pow(retryConfig.backoffMultiplier, attemptNumber)
          console.warn(`Retry attempt ${attemptNumber + 1}/${retryConfig.maxRetries} after ${delay}ms`)
          await sleep(delay)
          return executeRequest(attemptNumber + 1)
        }
        
        throw error
        
      } catch (error: any) {
        if (timeoutId) clearTimeout(timeoutId)
        
        if (error.name === 'AbortError') {
          throw new Error('Request timeout')
        }
        
        // Network errors - retry
        if (!error.code && attemptNumber < retryConfig.maxRetries) {
          const delay = retryConfig.retryDelay * Math.pow(retryConfig.backoffMultiplier, attemptNumber)
          console.warn(`Network error - retry ${attemptNumber + 1}/${retryConfig.maxRetries}`)
          await sleep(delay)
          return executeRequest(attemptNumber + 1)
        }
        
        pendingRequests.delete(cacheKey)
        throw error
      }
    }
    
    const promise = executeRequest()
    pendingRequests.set(cacheKey, promise)
    return promise
  }

  /**
   * Batch requests для оптимизации
   */
  async function batchRequest<T>(
    requests: Array<{ endpoint: string; options?: ApiRequestOptions }>
  ): Promise<Array<T | null>> {
    return Promise.all(
      requests.map(({ endpoint, options }) => 
        request<T>(endpoint, options).catch(err => {
          console.error(`Batch request failed for ${endpoint}:`, err)
          return null
        })
      )
    )
  }

  /**
   * Clear cache
   */
  function clearCache(pattern?: string) {
    if (!pattern) {
      requestCache.clear()
      return
    }
    
    const regex = new RegExp(pattern)
    for (const [key] of requestCache) {
      if (regex.test(key)) {
        requestCache.delete(key)
      }
    }
  }

  return {
    request,
    batchRequest,
    clearCache,
    token,
    refreshToken,
    isRefreshing: computed(() => isRefreshing.value),
    
    // HTTP method shortcuts
    get: <T>(endpoint: string, options?: ApiRequestOptions) => 
      request<T>(endpoint, { ...options, method: 'GET' }),
    post: <T>(endpoint: string, body: any, options?: ApiRequestOptions) => 
      request<T>(endpoint, { ...options, method: 'POST', body }),
    put: <T>(endpoint: string, body: any, options?: ApiRequestOptions) => 
      request<T>(endpoint, { ...options, method: 'PUT', body }),
    patch: <T>(endpoint: string, body: any, options?: ApiRequestOptions) => 
      request<T>(endpoint, { ...options, method: 'PATCH', body }),
    delete: <T>(endpoint: string, options?: ApiRequestOptions) => 
      request<T>(endpoint, { ...options, method: 'DELETE' })
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
