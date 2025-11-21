/**
 * useSystemStatus.ts — Composable для статуса системы
 * Typed API integration, авто-refresh, error handling
 * Enterprise: использует type guards вместо assertions
 */
import { ref } from 'vue'
import { useGeneratedApi } from './useGeneratedApi'
import { isErrorResponse, isSystemStatus } from '~/types/guards'
import type { SystemStatus, AsyncState, ErrorResponse } from '../types/api'

export function useSystemStatus(systemId: string, refreshInterval = 10000) {
  const { request } = useGeneratedApi()
  const state = ref<AsyncState<SystemStatus>>({ data: null, loading: false, error: null })
  let timer: ReturnType<typeof setTimeout> | null = null

  async function load() {
    state.value.loading = true
    state.value.error = null
    try {
      const resp = await request(`/systems/${systemId}/status`, { method: 'GET' })
      
      if (isErrorResponse(resp)) {
        state.value.error = resp
      } else if (isSystemStatus(resp)) {
        state.value.data = resp
      } else {
        throw new Error('Invalid response shape from API')
      }
    } catch (err) {
      state.value.error = { 
        message: String(err),
        code: 'NETWORK_ERROR',
        error: { message: String(err), code: 'NETWORK_ERROR', timestamp: '', request_id: '' }
      }
    } finally {
      state.value.loading = false
    }
  }

  /** Авто-refresh */
  function startAutoRefresh() {
    stopAutoRefresh()
    timer = setInterval(load, refreshInterval)
  }

  function stopAutoRefresh() {
    if (timer) clearInterval(timer)
    timer = null
  }

  // Immediate load (manually)
  load()
  startAutoRefresh()

  return {
    state,
    load,
    startAutoRefresh,
    stopAutoRefresh
  }
}
