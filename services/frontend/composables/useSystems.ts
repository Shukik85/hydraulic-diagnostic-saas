/**
 * Systems Management Composable
 * @module composables/useSystems
 * @description API integration for systems CRUD operations
 * Includes error handling, loading states, and accessibility helpers
 */

import { ref, computed, Ref } from 'vue'
import type {
  SystemSummary,
  SystemDetail,
  SystemSensor,
  SystemCreateInput,
  SystemUpdateInput,
  SystemFilterOptions,
  PaginatedSystemsResponse,
  SystemResponse,
  SensorReadingResponse,
  ErrorResponse,
} from '~/types/systems'

interface UseSystemsState {
  systems: Ref<SystemSummary[]>
  currentSystem: Ref<SystemDetail | null>
  sensors: Ref<SystemSensor[]>
  loading: Ref<boolean>
  error: Ref<string | null>
  pagination: Ref<{
    page: number
    pageSize: number
    total: number
    hasMore: boolean
  }>
}

export function useSystems(): {
  // State
  systems: Ref<SystemSummary[]>
  currentSystem: Ref<SystemDetail | null>
  sensors: Ref<SystemSensor[]>
  loading: Ref<boolean>
  error: Ref<string | null>
  pagination: Ref<{
    page: number
    pageSize: number
    total: number
    hasMore: boolean
  }>
  // Computed
  hasError: any
  isEmpty: any
  systemsCount: any
  // Methods
  fetchSystems: (filters?: SystemFilterOptions) => Promise<void>
  fetchSystemById: (systemId: string) => Promise<void>
  fetchSystemSensors: (systemId: string) => Promise<void>
  createSystem: (input: SystemCreateInput) => Promise<SystemDetail | null>
  updateSystem: (systemId: string, input: SystemUpdateInput) => Promise<void>
  deleteSystem: (systemId: string) => Promise<void>
  clearError: () => void
  clearCurrentSystem: () => void
} {
  // State
  const systems = ref<SystemSummary[]>([])
  const currentSystem = ref<SystemDetail | null>(null)
  const sensors = ref<SystemSensor[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const pagination = ref({
    page: 1,
    pageSize: 20,
    total: 0,
    hasMore: false,
  })

  const { $fetch } = useNuxtApp()
  const toast = useToast()

  // Computed
  const hasError = computed(() => error.value !== null)
  const isEmpty = computed(() => systems.value.length === 0)
  const systemsCount = computed(() => systems.value.length)

  /**
   * Fetch paginated systems list
   * @param filters - Optional filter options
   */
  const fetchSystems = async (filters?: SystemFilterOptions): Promise<void> => {
    loading.value = true
    error.value = null

    try {
      const params = new URLSearchParams()

      if (filters?.search) {
        params.append('search', filters.search)
      }
      if (filters?.status?.length) {
        params.append('status', filters.status.join(','))
      }
      if (filters?.equipmentType?.length) {
        params.append('type', filters.equipmentType.join(','))
      }
      if (filters?.sortBy) {
        params.append('sortBy', filters.sortBy)
      }
      if (filters?.sortOrder) {
        params.append('order', filters.sortOrder)
      }
      if (filters?.page) {
        params.append('page', filters.page.toString())
      }
      if (filters?.pageSize) {
        params.append('pageSize', filters.pageSize.toString())
      }

      const response = await $fetch<PaginatedSystemsResponse>(
        `/api/v1/systems?${params.toString()}`
      )

      if (response.status === 'success') {
        systems.value = response.data
        pagination.value = {
          page: response.page,
          pageSize: response.pageSize,
          total: response.total,
          hasMore: response.hasMore,
        }
      } else {
        throw new Error('Failed to fetch systems')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to fetch systems'
      error.value = message
      toast?.error(message)
    } finally {
      loading.value = false
    }
  }

  /**
   * Fetch complete system details by ID
   * @param systemId - System ID
   */
  const fetchSystemById = async (systemId: string): Promise<void> => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<SystemResponse>(
        `/api/v1/systems/${systemId}`
      )

      if (response.status === 'success') {
        currentSystem.value = response.data
      } else {
        throw new Error('Failed to fetch system details')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to fetch system details'
      error.value = message
      toast?.error(message)
    } finally {
      loading.value = false
    }
  }

  /**
   * Fetch real-time sensor readings for a system
   * @param systemId - System ID
   */
  const fetchSystemSensors = async (systemId: string): Promise<void> => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<SensorReadingResponse>(
        `/api/v1/systems/${systemId}/sensors`
      )

      if (response.status === 'success') {
        sensors.value = response.data
      } else {
        throw new Error('Failed to fetch sensor data')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to fetch sensor data'
      error.value = message
      toast?.error(message)
    } finally {
      loading.value = false
    }
  }

  /**
   * Create new system with topology
   * @param input - System creation input
   * @returns Created system details or null on error
   */
  const createSystem = async (
    input: SystemCreateInput
  ): Promise<SystemDetail | null> => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<any>('/api/v1/systems', {
        method: 'POST',
        body: input,
      })

      if (response.status === 'success') {
        toast?.success('System created successfully')
        // Refresh systems list
        await fetchSystems()
        return response.data
      } else {
        throw new Error('Failed to create system')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to create system'
      error.value = message
      toast?.error(message)
      return null
    } finally {
      loading.value = false
    }
  }

  /**
   * Update existing system
   * @param systemId - System ID
   * @param input - Update input
   */
  const updateSystem = async (
    systemId: string,
    input: SystemUpdateInput
  ): Promise<void> => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<SystemResponse>(
        `/api/v1/systems/${systemId}`,
        {
          method: 'PATCH',
          body: input,
        }
      )

      if (response.status === 'success') {
        currentSystem.value = response.data
        toast?.success('System updated successfully')
      } else {
        throw new Error('Failed to update system')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to update system'
      error.value = message
      toast?.error(message)
    } finally {
      loading.value = false
    }
  }

  /**
   * Delete system
   * @param systemId - System ID
   */
  const deleteSystem = async (systemId: string): Promise<void> => {
    loading.value = true
    error.value = null

    try {
      const response = await $fetch<{ status: string }>(
        `/api/v1/systems/${systemId}`,
        {
          method: 'DELETE',
        }
      )

      if (response.status === 'success') {
        systems.value = systems.value.filter((s) => s.systemId !== systemId)
        toast?.success('System deleted successfully')
      } else {
        throw new Error('Failed to delete system')
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to delete system'
      error.value = message
      toast?.error(message)
    } finally {
      loading.value = false
    }
  }

  /**
   * Clear error state
   */
  const clearError = (): void => {
    error.value = null
  }

  /**
   * Clear current system and sensors
   */
  const clearCurrentSystem = (): void => {
    currentSystem.value = null
    sensors.value = []
  }

  return {
    // State
    systems,
    currentSystem,
    sensors,
    loading,
    error,
    pagination,
    // Computed
    hasError,
    isEmpty,
    systemsCount,
    // Methods
    fetchSystems,
    fetchSystemById,
    fetchSystemSensors,
    createSystem,
    updateSystem,
    deleteSystem,
    clearError,
    clearCurrentSystem,
  }
}
