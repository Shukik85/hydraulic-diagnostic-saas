/**
 * Systems Pinia Store
 * @module stores/systems
 * @description Global state management for systems domain
 * Handles cross-component system state with reactive updates
 */

import { defineStore } from 'pinia'
import type {
  SystemSummary,
  SystemDetail,
  SystemSensor,
  SystemFilterOptions,
  SensorFilterOptions,
} from '~/types/systems'

interface SystemsState {
  systems: SystemSummary[]
  currentSystem: SystemDetail | null
  sensors: SystemSensor[]
  loading: boolean
  error: string | null
  filters: SystemFilterOptions
  sensorFilters: SensorFilterOptions
  pagination: {
    page: number
    pageSize: number
    total: number
    hasMore: boolean
  }
  lastRefresh: number
}

export const useSystemsStore = defineStore('systems', {
  state: (): SystemsState => ({
    systems: [],
    currentSystem: null,
    sensors: [],
    loading: false,
    error: null,
    filters: {
      sortBy: 'created',
      sortOrder: 'desc',
      page: 1,
      pageSize: 20,
    },
    sensorFilters: {
      sortBy: 'updated',
      sortOrder: 'desc',
    },
    pagination: {
      page: 1,
      pageSize: 20,
      total: 0,
      hasMore: false,
    },
    lastRefresh: 0,
  }),

  getters: {
    /**
     * Get system by ID
     */
    getSystemById: (state) => (systemId: string): SystemSummary | undefined => {
      return state.systems.find((s) => s.systemId === systemId)
    },

    /**
     * Get filtered systems
     * Applies current filters to systems list
     */
    getFilteredSystems: (state) => (): SystemSummary[] => {
      let filtered = [...state.systems]

      if (state.filters.search) {
        const search = state.filters.search.toLowerCase()
        filtered = filtered.filter(
          (s) =>
            s.equipmentName.toLowerCase().includes(search) ||
            s.equipmentId.toLowerCase().includes(search) ||
            s.equipmentType.toLowerCase().includes(search)
        )
      }

      if (state.filters.status?.length) {
        filtered = filtered.filter((s) => state.filters.status?.includes(s.status))
      }

      if (state.filters.equipmentType?.length) {
        filtered = filtered.filter((s) =>
          state.filters.equipmentType?.includes(s.equipmentType)
        )
      }

      // Sort
      const sortBy = state.filters.sortBy || 'created'
      const isAsc = state.filters.sortOrder === 'asc'

      filtered.sort((a, b) => {
        let aVal: any = a[sortBy as keyof SystemSummary]
        let bVal: any = b[sortBy as keyof SystemSummary]

        if (typeof aVal === 'string') {
          aVal = aVal.toLowerCase()
          bVal = (bVal as string).toLowerCase()
        }

        if (aVal < bVal) return isAsc ? -1 : 1
        if (aVal > bVal) return isAsc ? 1 : -1
        return 0
      })

      return filtered
    },

    /**
     * Get filtered sensors
     */
    getFilteredSensors: (state) => (): SystemSensor[] => {
      let filtered = [...state.sensors]

      if (state.sensorFilters.sensorType?.length) {
        filtered = filtered.filter((s) =>
          state.sensorFilters.sensorType?.includes(s.sensorType)
        )
      }

      if (state.sensorFilters.status?.length) {
        filtered = filtered.filter((s) =>
          state.sensorFilters.status?.includes(s.status)
        )
      }

      if (state.sensorFilters.location?.length) {
        filtered = filtered.filter((s) =>
          state.sensorFilters.location?.includes(s.componentId)
        )
      }

      // Sort
      const sortBy = state.sensorFilters.sortBy || 'updated'
      const isAsc = state.sensorFilters.sortOrder === 'asc'

      filtered.sort((a, b) => {
        let aVal: any = a[sortBy as keyof SystemSensor]
        let bVal: any = b[sortBy as keyof SystemSensor]

        if (typeof aVal === 'string') {
          aVal = aVal.toLowerCase()
          bVal = (bVal as string).toLowerCase()
        }

        if (aVal < bVal) return isAsc ? -1 : 1
        if (aVal > bVal) return isAsc ? 1 : -1
        return 0
      })

      return filtered
    },

    /**
     * Get online systems count
     */
    getOnlineSystemsCount: (state) => (): number => {
      return state.systems.filter((s) => s.status === 'online').length
    },

    /**
     * Get systems with warnings
     */
    getSystemsWithWarnings: (state) => (): SystemSummary[] => {
      return state.systems.filter((s) => s.status === 'degraded')
    },

    /**
     * Get offline systems
     */
    getOfflineSystems: (state) => (): SystemSummary[] => {
      return state.systems.filter((s) => s.status === 'offline')
    },

    /**
     * Check if should refresh (cache expiry)
     * Default: 5 minutes
     */
    shouldRefresh: (state) => (): boolean => {
      const now = Date.now()
      const cacheExpiry = 5 * 60 * 1000 // 5 minutes
      return now - state.lastRefresh > cacheExpiry
    },

    /**
     * Get sensor status summary
     */
    getSensorStatusSummary: (state) => () => {
      const summary = {
        ok: 0,
        warning: 0,
        error: 0,
        offline: 0,
      }

      state.sensors.forEach((sensor) => {
        summary[sensor.status] = (summary[sensor.status] || 0) + 1
      })

      return summary
    },
  },

  actions: {
    /**
     * Set systems list
     */
    setSystems(systems: SystemSummary[]): void {
      this.systems = systems
    },

    /**
     * Add or update system in list
     */
    upsertSystem(system: SystemSummary): void {
      const index = this.systems.findIndex((s) => s.systemId === system.systemId)
      if (index >= 0) {
        this.systems[index] = system
      } else {
        this.systems.unshift(system)
      }
    },

    /**
     * Remove system from list
     */
    removeSystem(systemId: string): void {
      this.systems = this.systems.filter((s) => s.systemId !== systemId)
    },

    /**
     * Set current system details
     */
    setCurrentSystem(system: SystemDetail | null): void {
      this.currentSystem = system
    },

    /**
     * Set sensor readings
     */
    setSensors(sensors: SystemSensor[]): void {
      this.sensors = sensors
    },

    /**
     * Update single sensor reading
     */
    updateSensorReading(sensorId: string, reading: Partial<SystemSensor>): void {
      const index = this.sensors.findIndex((s) => s.sensorId === sensorId)
      if (index >= 0) {
        this.sensors[index] = { ...this.sensors[index], ...reading }
      }
    },

    /**
     * Set loading state
     */
    setLoading(loading: boolean): void {
      this.loading = loading
    },

    /**
     * Set error state
     */
    setError(error: string | null): void {
      this.error = error
    },

    /**
     * Clear error
     */
    clearError(): void {
      this.error = null
    },

    /**
     * Update filters
     */
    updateFilters(filters: Partial<SystemFilterOptions>): void {
      this.filters = { ...this.filters, ...filters }
    },

    /**
     * Reset filters to defaults
     */
    resetFilters(): void {
      this.filters = {
        sortBy: 'created',
        sortOrder: 'desc',
        page: 1,
        pageSize: 20,
      }
    },

    /**
     * Update sensor filters
     */
    updateSensorFilters(filters: Partial<SensorFilterOptions>): void {
      this.sensorFilters = { ...this.sensorFilters, ...filters }
    },

    /**
     * Reset sensor filters
     */
    resetSensorFilters(): void {
      this.sensorFilters = {
        sortBy: 'updated',
        sortOrder: 'desc',
      }
    },

    /**
     * Update pagination state
     */
    updatePagination(data: {
      page: number
      pageSize: number
      total: number
      hasMore: boolean
    }): void {
      this.pagination = data
    },

    /**
     * Update last refresh timestamp
     */
    updateLastRefresh(): void {
      this.lastRefresh = Date.now()
    },
  },
})
