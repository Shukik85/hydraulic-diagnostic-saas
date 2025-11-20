/**
 * useSystemStatus.ts — Composable для статуса системы
 * Typed API integration, авто-refresh, error handling
 */
import { ref } from 'vue'
import { useGeneratedApi } from './useGeneratedApi'
import type { SystemStatus, AsyncState, ErrorResponse } from '../types/api'

export function useSystemStatus(systemId: string, refreshInterval = 10000) {
  const { request } = useGeneratedApi()
  const state = ref<AsyncState<SystemStatus>>({ data: null, loading: false, error: null })
  let timer: ReturnType<typeof setTimeout> | null = null

  async function load() {
    state.value.loading = true
    state.value.error = null
    try {
      const resp = await request(`/systems/${systemId}/status`, { method: 'GET' }) as SystemStatus | ErrorResponse
      
      if (resp && 'data' in resp) {
        state.value.data = resp as SystemStatus
      } else if (resp && 'error' in resp) {
        state.value.error = resp as ErrorResponse
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
