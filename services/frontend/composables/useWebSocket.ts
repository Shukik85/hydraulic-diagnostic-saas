/**
 * useWebSocket.ts — реактивный WebSocket composable
 * Поддержка real-time sensor data, anomaly alerts, system status updates
 * Типизировано по OpenAPI v3.1 спецификации
 */
import { ref, onUnmounted, computed } from 'vue'
import type {
  WSMessage,
  WSNewSensorReading,
  WSNewAnomaly,
  WSSystemStatusUpdate,
  isValidWSMessage
} from '../types/api'

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
 * WebSocket composable для real-time коммуникации
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
  
  function handleOpen() {
    log('Connected')
    connectionState.value = WSConnectionState.Connected
    reconnectAttempts = 0
    lastError.value = null
    startHeartbeat()
  }
  
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
          try { h(message.data) } catch (e) { logError('Handler error:', e) }
        })
      }
    } catch (error) {
      logError('Parse error:', error)
    }
  }
  
  function handleError(event: Event) {
    logError('Error:', event)
    lastError.value = event
    connectionState.value = WSConnectionState.Error
  }
  
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
  
  function on<T extends WSMessage['type']>(
    type: T,
    handler: (data: Extract<WSMessage, { type: T }>['data']) => void
  ): () => void {
    if (!messageHandlers.has(type)) {
      messageHandlers.set(type, new Set())
    }
    
    const handlers = messageHandlers.get(type)!
    handlers.add(handler)
    
    return () => handlers.delete(handler)
  }
  
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
  
  function onSensorReading(handler: (data: WSNewSensorReading['data']) => void) {
    return on('sensor_reading', handler)
  }
  
  function onAnomalyDetected(handler: (data: WSNewAnomaly['data']) => void) {
    return on('anomaly_detected', handler)
  }
  
  function onSystemStatusUpdate(handler: (data: WSSystemStatusUpdate['data']) => void) {
    return on('system_status_update', handler)
  }
  
  onUnmounted(() => {
    log('Cleanup')
    disconnect()
  })
  
  return {
    connectionState,
    isConnected,
    lastMessage,
    lastError,
    connect,
    disconnect,
    send,
    on,
    onSensorReading,
    onAnomalyDetected,
    onSystemStatusUpdate
  }
}

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
