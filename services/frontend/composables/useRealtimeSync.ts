/**
 * useRealtimeSync.ts — Синхронизация REST API и WebSocket
 * 
 * Features:
 * - Automatic fallback to polling when WebSocket disconnected
 * - State synchronization between REST and real-time updates
 * - Toast notifications for critical events
 * - Optimistic UI updates
 */
import { ref, watch, onUnmounted } from 'vue'
import { useWebSocketAdvanced } from './useWebSocketAdvanced'
import { useMetadataStore } from '~/stores/metadata'
import { useSystemsStore } from '~/stores/systems.store'

export interface RealtimeSyncOptions {
  pollingInterval?: number // ms, default 10000
  enableNotifications?: boolean // default true
  autoReconnect?: boolean // default true
}

const DEFAULT_OPTIONS: RealtimeSyncOptions = {
  pollingInterval: 10000,
  enableNotifications: true,
  autoReconnect: true
}

/**
 * Real-time synchronization composable
 */
export function useRealtimeSync(options: RealtimeSyncOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options }
  
  const ws = useWebSocketAdvanced({
    autoReconnect: opts.autoReconnect
  })
  
  const metadataStore = useMetadataStore()
  const systemsStore = useSystemsStore()
  
  const toast = useToast()
  const pollInterval = ref<ReturnType<typeof setInterval> | null>(null)
  const isSyncing = ref(false)
  
  /**
   * Sync sensor readings from WebSocket
   */
  ws.onSensorReading((data) => {
    const { system_id, component_id, sensor_type, value, timestamp, unit } = data
    
    // Update metadata store optimistically
    const component = metadataStore.componentsMap.value.get(component_id)
    if (component) {
      if (!component.latest_readings) {
        component.latest_readings = {}
      }
      component.latest_readings[sensor_type] = { 
        value, 
        timestamp, 
        unit: unit || 'bar' 
      }
    }
    
    // Update system status if needed
    if (system_id && systemsStore.state.value.currentSystem?.id === system_id) {
      // Trigger reactivity
      systemsStore.state.value.currentSystem = { 
        ...systemsStore.state.value.currentSystem 
      }
    }
  })
  
  /**
   * Sync anomaly detections
   */
  ws.onAnomalyDetected((data) => {
    const { system_id, component_id, severity, description, detected_at } = data
    
    // Add to systems store alerts
    if (systemsStore.addAlert) {
      systemsStore.addAlert({
        id: crypto.randomUUID(),
        system_id,
        component_id,
        severity,
        message: description,
        timestamp: detected_at || Date.now(),
        acknowledged: false
      })
    }
    
    // Show toast notification for critical anomalies
    if (opts.enableNotifications && (severity === 'critical' || severity === 'high')) {
      toast.add({
        title: severity === 'critical' ? 'Критическая аномалия' : 'Высокая аномалия',
        description: description || 'Обнаружено отклонение от нормы',
        color: severity === 'critical' ? 'red' : 'orange',
        timeout: severity === 'critical' ? 0 : 8000, // Critical stays visible
        actions: [{
          label: 'Посмотреть',
          click: () => {
            if (system_id) {
              navigateTo(`/systems/${system_id}`)
            }
          }
        }]
      })
    }
  })
  
  /**
   * Sync system status updates
   */
  ws.onSystemStatusUpdate((data) => {
    const { system_id, status, health_score, last_diagnostic_at } = data
    
    // Update system in store
    const system = systemsStore.state.value.systems.find(s => s.id === system_id)
    if (system) {
      system.status = status
      system.health_score = health_score
      system.last_diagnostic_at = last_diagnostic_at
    }
    
    // Update current system if viewing
    if (systemsStore.state.value.currentSystem?.id === system_id) {
      systemsStore.state.value.currentSystem = {
        ...systemsStore.state.value.currentSystem,
        status,
        health_score,
        last_diagnostic_at
      }
    }
  })
  
  /**
   * Fallback polling when WebSocket disconnected
   */
  async function startPolling() {
    if (pollInterval.value) return
    
    console.log('[RealtimeSync] Starting fallback polling')
    
    pollInterval.value = setInterval(async () => {
      if (isSyncing.value) return
      
      isSyncing.value = true
      try {
        // Fetch latest data from REST API
        await Promise.allSettled([
          metadataStore.fetchLatestReadings?.(),
          systemsStore.refreshCurrentSystem?.()
        ])
      } catch (error) {
        console.error('[RealtimeSync] Polling failed:', error)
      } finally {
        isSyncing.value = false
      }
    }, opts.pollingInterval)
  }
  
  function stopPolling() {
    if (pollInterval.value) {
      console.log('[RealtimeSync] Stopping fallback polling')
      clearInterval(pollInterval.value)
      pollInterval.value = null
    }
  }
  
  /**
   * Watch WebSocket connection state
   * Start polling if disconnected, stop when reconnected
   */
  watch(() => ws.isConnected.value, (connected) => {
    if (!connected) {
      startPolling()
      
      if (opts.enableNotifications) {
        toast.add({
          title: 'Подключение потеряно',
          description: 'Переключение на периодическое обновление',
          color: 'yellow',
          timeout: 5000
        })
      }
    } else {
      stopPolling()
      
      if (opts.enableNotifications) {
        toast.add({
          title: 'Подключение восстановлено',
          description: 'Реальное время восстановлено',
          color: 'green',
          timeout: 3000
        })
      }
    }
  })
  
  /**
   * Initialize connection
   */
  function connect() {
    ws.connect()
  }
  
  /**
   * Disconnect and cleanup
   */
  function disconnect() {
    stopPolling()
    ws.disconnect()
  }
  
  /**
   * Manual sync trigger
   */
  async function syncNow() {
    if (isSyncing.value) return
    
    isSyncing.value = true
    try {
      await Promise.all([
        metadataStore.fetchLatestReadings?.(),
        systemsStore.refreshCurrentSystem?.()
      ])
      
      if (opts.enableNotifications) {
        toast.add({
          title: 'Данные обновлены',
          color: 'green',
          timeout: 2000
        })
      }
    } catch (error) {
      console.error('[RealtimeSync] Manual sync failed:', error)
      throw error
    } finally {
      isSyncing.value = false
    }
  }
  
  onUnmounted(() => {
    disconnect()
  })
  
  return {
    // WebSocket state
    isConnected: ws.isConnected,
    connectionState: ws.connectionState,
    connectionHealth: ws.connectionHealth,
    metrics: ws.metrics,
    statistics: ws.statistics,
    
    // Sync state
    isSyncing,
    isPolling: computed(() => pollInterval.value !== null),
    
    // Actions
    connect,
    disconnect,
    syncNow,
    
    // Utilities
    resetMetrics: ws.resetMetrics
  }
}

/**
 * Auto-start real-time sync on app mount
 */
export function useAutoRealtimeSync(options?: RealtimeSyncOptions) {
  const sync = useRealtimeSync(options)
  
  onMounted(() => {
    sync.connect()
  })
  
  return sync
}
