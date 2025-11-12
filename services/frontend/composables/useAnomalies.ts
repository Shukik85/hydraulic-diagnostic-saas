/**
 * useAnomalies - Composable for anomaly detection operations
 * 
 * Features:
 * - Typed API integration
 * - Auto-loading with reactive filters
 * - Pagination support
 * - Error handling
 * 
 * @example
 * const { state, page, severity, load } = useAnomalies('system-uuid')
 */
import { ref, watchEffect } from 'vue'
import { useApi } from './useApi'
import type {
  AnomaliesQueryParams,
  AnomaliesListResponse,
  AsyncState,
  ErrorResponse,
  AnomalySeverity,
  ErrorCode
} from '../types/api'

export function useAnomalies(systemId: string, filters: Partial<AnomaliesQueryParams> = {}) {
  const { request } = useApi()
  
  // Reactive state
  const state = ref<AsyncState<AnomaliesListResponse>>({
    data: null,
    loading: false,
    error: null
  })
  
  // Pagination
  const page = ref(filters.page || 1)
  const perPage = ref(filters.per_page || 20)
  
  // Filters (reactive)
  const severity = ref<AnomalySeverity | undefined>(filters.severity)
  const startDate = ref<string | undefined>(filters.start_date)
  const endDate = ref<string | undefined>(filters.end_date)

  /**
   * Load anomalies from API
   */
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
      
      const resp = await request<AnomaliesListResponse>(`/systems/${systemId}/anomalies`, {
        method: 'GET',
        params
      })
      
      if ('data' in resp) {
        state.value.data = resp.data
      } else {
        state.value.error = resp as ErrorResponse
      }
    } catch (err) {
      // Network or unexpected errors
      state.value.error = {
        error: {
          message: String(err),
          code: ErrorCode.NetworkError,
          timestamp: new Date().toISOString(),
          request_id: ''
        }
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
