/**
 * useWebSocket - Reactive WebSocket composable
 * 
 * Features:
 * - Real-time sensor data streaming
 * - Anomaly alerts
 * - System status updates
 * - Auto-reconnection
 * - Type-safe message handling
 * 
 * @example
 * const ws = useWebSocket()
 * ws.connect()
 * ws.onAnomalyDetected((data) => console.log(data))
 */
import { ref, onUnmounted, computed } from 'vue'
import type {
  WSMessage,
  WSNewSensorReading,
  WSNewAnomaly,
  WSSystemStatusUpdate
} from '../types/api'
import { isValidWSMessage } from '../types/api'

/**
 * WebSocket connection states
 */
export enum WSConnectionState {
  Disconnected = 'disconnected',
  Connecting = 'connecting',
  Connected = 'connected',
  Reconnecting = 'reconnecting',
  Error = 'error'
}

/**
 * WebSocket composable options
 */
export interface UseWebSocketOptions {
  url?: string
  autoReconnect?: boolean
  reconnectDelay?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
  debug?: boolean
}

const DEFAULT_OPTIONS: Required<UseWebSocketOptions> = {
  url: 'ws://localhost:8000/ws',
  autoReconnect: true,
  reconnectDelay: 3000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000,
  debug: false
}

/**
 * WebSocket composable for real-time communication
 */
export function useWebSocket(options: UseWebSocketOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options }
  
  const ws = ref<WebSocket | null>(null)
  const connectionState = ref<WSConnectionState>(WSConnectionState.Disconnected)
  const isConnected = computed(() => connectionState.value === WSConnectionState.Connected)
  const lastMessage = ref<WSMessage | null>(null)
  const lastError = ref<Event | null>(null)
  
  let reconnectAttempts = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  
  const messageHandlers = new Map<string, Set<(data: any) => void>>()
  
  const log = (...args: any[]) => {
    if (opts.debug) console.log('[WebSocket]', ...args)
  }
  
  const logError = (...args: any[]) => {
    console.error('[WebSocket ERROR]', ...args)
  }
  
  /**
   * Connect to WebSocket server
   */
  function connect() {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      log('Already connected')
      return
    }
    
    try {
      log('Connecting to', opts.url)
      connectionState.value = reconnectAttempts > 0 
        ? WSConnectionState.Reconnecting 
        : WSConnectionState.Connecting
      
      ws.value = new WebSocket(opts.url)
      
      ws.value.onopen = handleOpen
      ws.value.onmessage = handleMessage
      ws.value.onerror = handleError
      ws.value.onclose = handleClose
      
    } catch (error) {
      logError('Connection failed:', error)
      connectionState.value = WSConnectionState.Error
      scheduleReconnect()
    }
  }
  
  /**
   * Disconnect from WebSocket server
   */
  function disconnect() {
    log('Disconnecting...')
    
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer)
      heartbeatTimer = null
    }
    
    if (ws.value) {
      ws.value.close(1000, 'Client disconnect')
      ws.value = null
    }
    
    connectionState.value = WSConnectionState.Disconnected
    reconnectAttempts = 0
  }
  
  /**
   * Schedule reconnection attempt
   */
  function scheduleReconnect() {
    if (!opts.autoReconnect) return
    
    if (reconnectAttempts >= opts.maxReconnectAttempts) {
      logError(`Max reconnect attempts reached`)
      connectionState.value = WSConnectionState.Error
      return
    }
    
    reconnectAttempts++
    log(`Reconnect attempt ${reconnectAttempts}/${opts.maxReconnectAttempts}`)
    
    reconnectTimer = setTimeout(() => connect(), opts.reconnectDelay)
  }
  
  /**
   * Handle WebSocket open event
   */
  function handleOpen() {
    log('Connected')
    connectionState.value = WSConnectionState.Connected
    reconnectAttempts = 0
    lastError.value = null
    startHeartbeat()
  }
  
  /**
   * Handle incoming WebSocket message
   */
  function handleMessage(event: MessageEvent) {
    try {
      const message = JSON.parse(event.data) as WSMessage
      
      if (!isValidWSMessage(message)) {
        logError('Invalid message format')
        return
      }
      
      log('Received:', message.type)
      lastMessage.value = message
      
      const handlers = messageHandlers.get(message.type)
      if (handlers) {
        handlers.forEach(h => {
          try {
            h(message.data)
          } catch (e) {
            logError('Handler error:', e)
          }
        })
      }
    } catch (error) {
      logError('Parse error:', error)
    }
  }
  
  /**
   * Handle WebSocket error
   */
  function handleError(event: Event) {
    logError('Error:', event)
    lastError.value = event
    connectionState.value = WSConnectionState.Error
  }
  
  /**
   * Handle WebSocket close
   */
  function handleClose(event: CloseEvent) {
    log('Closed:', event.code)
    
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer)
      heartbeatTimer = null
    }
    
    connectionState.value = WSConnectionState.Disconnected
    
    if (event.code !== 1000 && opts.autoReconnect) {
      scheduleReconnect()
    }
  }
  
  /**
   * Start heartbeat timer
   */
  function startHeartbeat() {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    
    heartbeatTimer = setInterval(() => {
      if (ws.value && ws.value.readyState === WebSocket.OPEN) {
        try {
          ws.value.send(JSON.stringify({ type: 'ping' }))
        } catch (e) {
          logError('Heartbeat failed:', e)
        }
      }
    }, opts.heartbeatInterval)
  }
  
  /**
   * Subscribe to specific message type
   */
  function on<T extends WSMessage['type']>(
    type: T,
    handler: (data: Extract<WSMessage, { type: T }>['data']) => void
  ): () => void {
    if (!messageHandlers.has(type)) {
      messageHandlers.set(type, new Set())
    }
    
    const handlers = messageHandlers.get(type)!
    handlers.add(handler)
    
    // Return unsubscribe function
    return () => handlers.delete(handler)
  }
  
  /**
   * Send message to server
   */
  function send(data: any): boolean {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      logError('Cannot send: not connected')
      return false
    }
    
    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      ws.value.send(message)
      return true
    } catch (error) {
      logError('Send failed:', error)
      return false
    }
  }
  
  /**
   * Subscribe to sensor readings
   */
  function onSensorReading(handler: (data: WSNewSensorReading['data']) => void) {
    return on('sensor_reading', handler)
  }
  
  /**
   * Subscribe to anomaly alerts
   */
  function onAnomalyDetected(handler: (data: WSNewAnomaly['data']) => void) {
    return on('anomaly_detected', handler)
  }
  
  /**
   * Subscribe to system status updates
   */
  function onSystemStatusUpdate(handler: (data: WSSystemStatusUpdate['data']) => void) {
    return on('system_status_update', handler)
  }
  
  // Cleanup on unmount
  onUnmounted(() => {
    log('Cleanup')
    disconnect()
  })
  
  return {
    // State
    connectionState,
    isConnected,
    lastMessage,
    lastError,
    
    // Methods
    connect,
    disconnect,
    send,
    on,
    
    // Convenience methods
    onSensorReading,
    onAnomalyDetected,
    onSystemStatusUpdate
  }
}

/**
 * Get Tailwind color classes for connection state
 */
export function getConnectionStateColor(state: WSConnectionState): string {
  const colors: Record<WSConnectionState, string> = {
    [WSConnectionState.Disconnected]: 'text-gray-500 bg-gray-100',
    [WSConnectionState.Connecting]: 'text-blue-500 bg-blue-100',
    [WSConnectionState.Connected]: 'text-green-500 bg-green-100',
    [WSConnectionState.Reconnecting]: 'text-yellow-500 bg-yellow-100',
    [WSConnectionState.Error]: 'text-red-500 bg-red-100'
  }
  return colors[state] || 'text-gray-500 bg-gray-100'
}

/**
 * Get icon name for connection state
 */
export function getConnectionStateIcon(state: WSConnectionState): string {
  const icons: Record<WSConnectionState, string> = {
    [WSConnectionState.Disconnected]: 'heroicons:x-circle',
    [WSConnectionState.Connecting]: 'heroicons:arrow-path',
    [WSConnectionState.Connected]: 'heroicons:check-circle',
    [WSConnectionState.Reconnecting]: 'heroicons:arrow-path',
    [WSConnectionState.Error]: 'heroicons:exclamation-circle'
  }
  return icons[state] || 'heroicons:question-mark-circle'
}
