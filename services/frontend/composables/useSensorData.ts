/**
 * Real-time Sensor Data Composable
 * @module composables/useSensorData
 * @description WebSocket-based sensor data streaming with fallback polling
 * Includes automatic reconnection, error handling, and memory management
 */

import { ref, computed, watch, Ref } from 'vue'
import type { SystemSensor, SensorStatus } from '~/types/systems'

interface UseSensorDataOptions {
  systemId: string
  pollingInterval?: number // ms, default 5000
  maxRetries?: number
  autoConnect?: boolean
}

interface SensorDataState {
  sensors: Ref<SystemSensor[]>
  isConnected: Ref<boolean>
  isLoading: Ref<boolean>
  error: Ref<string | null>
  lastUpdate: Ref<number>
  retryCount: Ref<number>
}

export function useSensorData(options: UseSensorDataOptions) {
  const {
    systemId,
    pollingInterval = 5000,
    maxRetries = 3,
    autoConnect = true,
  } = options

  const sensors = ref<SystemSensor[]>([])
  const isConnected = ref(false)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const lastUpdate = ref(0)
  const retryCount = ref(0)

  const { $fetch } = useNuxtApp()

  // WebSocket instance (if using real-time)
  let ws: WebSocket | null = null
  let pollInterval: NodeJS.Timeout | null = null

  /**
   * Fetch sensor data from API
   */
  const fetchSensorData = async (): Promise<void> => {
    isLoading.value = true
    error.value = null

    try {
      const response = await $fetch<any>(`/api/v1/systems/${systemId}/sensors`)

      if (response.status === 'success') {
        sensors.value = response.data
        lastUpdate.value = Date.now()
        retryCount.value = 0
      } else {
        throw new Error('Failed to fetch sensor data')
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sensor data'
      error.value = message

      if (retryCount.value < maxRetries) {
        retryCount.value++
        // Exponential backoff
        const delay = Math.pow(2, retryCount.value) * 1000
        setTimeout(fetchSensorData, delay)
      }
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Connect to WebSocket for real-time updates
   * Falls back to polling if WebSocket unavailable
   */
  const connect = async (): Promise<void> => {
    // First, fetch initial data
    await fetchSensorData()

    // Try to establish WebSocket connection
    const wsUrl = `${useRuntimeConfig().public.wsUrl}/systems/${systemId}/sensors`

    try {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        isConnected.value = true
        error.value = null
        retryCount.value = 0
      }

      ws.onmessage = (event: MessageEvent) => {
        try {
          const update = JSON.parse(event.data)
          if (update.type === 'sensor_update') {
            updateSensorReading(update.sensorId, update.reading)
            lastUpdate.value = Date.now()
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onerror = () => {
        isConnected.value = false
        error.value = 'WebSocket connection error'
        // Fall back to polling
        startPolling()
      }

      ws.onclose = () => {
        isConnected.value = false
        // Try to reconnect
        if (retryCount.value < maxRetries) {
          retryCount.value++
          const delay = Math.pow(2, retryCount.value) * 1000
          setTimeout(connect, delay)
        } else {
          startPolling()
        }
      }
    } catch (err) {
      // WebSocket not available, fall back to polling
      console.warn('WebSocket not available, using polling', err)
      startPolling()
    }
  }

  /**
   * Start polling for sensor data
   */
  const startPolling = (): void => {
    if (pollInterval) return // Already polling

    pollInterval = setInterval(async () => {
      await fetchSensorData()
    }, pollingInterval)
  }

  /**
   * Stop polling
   */
  const stopPolling = (): void => {
    if (pollInterval) {
      clearInterval(pollInterval)
      pollInterval = null
    }
  }

  /**
   * Update single sensor reading
   */
  const updateSensorReading = (sensorId: string, reading: Partial<SystemSensor>): void => {
    const index = sensors.value.findIndex((s) => s.sensorId === sensorId)
    if (index >= 0) {
      sensors.value[index] = {
        ...sensors.value[index],
        ...reading,
      }
    }
  }

  /**
   * Disconnect and cleanup
   */
  const disconnect = (): void => {
    stopPolling()

    if (ws) {
      ws.close()
      ws = null
    }

    isConnected.value = false
  }

  // Computed
  const sensorsSummary = computed(() => {
    const summary = {
      total: sensors.value.length,
      ok: 0,
      warning: 0,
      error: 0,
      offline: 0,
    }

    sensors.value.forEach((sensor) => {
      summary[sensor.status as SensorStatus]++
    })

    return summary
  })

  const errorSensors = computed(() => {
    return sensors.value.filter((s) => s.status === 'error')
  })

  const warningSensors = computed(() => {
    return sensors.value.filter((s) => s.status === 'warning')
  })

  // Auto-connect
  onMounted(() => {
    if (autoConnect) {
      connect()
    }
  })

  // Auto-disconnect
  onUnmounted(() => {
    disconnect()
  })

  // Watch for systemId changes
  watch(
    () => systemId,
    () => {
      disconnect()
      if (autoConnect) {
        connect()
      }
    }
  )

  return {
    // State
    sensors,
    isConnected,
    isLoading,
    error,
    lastUpdate,
    retryCount,
    // Computed
    sensorsSummary,
    errorSensors,
    warningSensors,
    // Methods
    connect,
    disconnect,
    fetchSensorData,
    startPolling,
    stopPolling,
    updateSensorReading,
  }
}
