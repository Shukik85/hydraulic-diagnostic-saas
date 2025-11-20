/**
 * useAnomalies.ts — Composable для работы с аномалиями
 * Typed API integration, авто-loading, пагинация, фильтры
 */
import { ref, watchEffect } from 'vue'
import { useGeneratedApi } from './useGeneratedApi'
import type {
  AnomaliesQueryParams,
  AnomaliesListResponse,
  AsyncState,
  ErrorResponse,
  AnomalySeverity
} from '../types/api'

export function useAnomalies(systemId: string, filters: Partial<AnomaliesQueryParams> = {}) {
  const { request } = useGeneratedApi()
  const state = ref<AsyncState<AnomaliesListResponse>>({ data: null, loading: false, error: null })
  const page = ref(filters.page || 1)
  const perPage = ref(filters.per_page || 20)
  // External filters (reactive)
  const severity = ref<AnomalySeverity | undefined>(filters.severity)
  const startDate = ref<string | undefined>(filters.start_date)
  const endDate = ref<string | undefined>(filters.end_date)

  // Load anomalies
  async function load() {
    state.value.loading = true
    state.value.error = null
    try {
      const params: AnomaliesQueryParams = {
        page: page.value,
        per_page: perPage.value,
        severity: severity.value,
        start_date: startDate.value,
        end_date: endDate.value
      }
      const resp = await request(`/systems/${systemId}/anomalies`, {
        method: 'GET',
        params
      }) as AnomaliesListResponse | ErrorResponse
      
      if (resp && 'data' in resp) {
        state.value.data = resp as AnomaliesListResponse
      } else {
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

  // Auto-load on params change
  watchEffect(() => {
    load()
  })

  return {
    state,
    page,
    perPage,
    severity,
    startDate,
    endDate,
    load
  }
}
