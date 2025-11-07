/**
 * useApi.ts — универсальный API client для Hydraulic Diagnostic Platform
 * Типизировано по OpenAPI v3.1.0
 */
import { ref } from 'vue'
import { useStorage } from '@vueuse/core'
import type {
  ApiRequestOptions,
  ApiResponse,
  ErrorResponse,
  isErrorResponse,
  getErrorMessage
} from '../types/api'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export function useApi() {
  // Access tokens храним в localStorage (JWT)
  const token = useStorage<string | null>('auth_token', null)
  
  async function request<T>(
    endpoint: string,
    options: ApiRequestOptions & { method?: 'GET'|'POST'|'PUT'|'PATCH'|'DELETE', body?: any } = {}
  ): Promise<ApiResponse<T> | ErrorResponse> {
    const url = endpoint.startsWith('http') ? endpoint : API_URL + endpoint
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers
    }
    if (token.value) headers['Authorization'] = `Bearer ${token.value}`

    try {
      const init: RequestInit = {
        method: options.method || 'GET',
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: options.signal
      }
      const res = await fetch(url, init)
      const text = await res.text()
      let data
      try { data = text ? JSON.parse(text) : null } catch { data = text }
      if (res.ok) {
        return { data, status: res.status, headers: parseHeaders(res.headers) }
      } else {
        return { error: data.error || { message: text, code: res.status }, status: res.status }
      }
    } catch (err) {
      return { error: { message: getErrorMessage(err), code: 'NETWORK_ERROR', timestamp: '', request_id: '' }, status: 0 }
    }
  }

  function parseHeaders(headers: Headers): Record<string, string> {
    const result: Record<string, string> = {}
    headers.forEach((v, k) => { result[k] = v })
    return result
  }

  return {
    request,
    token
  }
}
