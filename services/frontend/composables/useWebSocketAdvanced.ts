/**
 * useWebSocketAdvanced.ts — WebSocket с метриками и latency tracking
 * Расширяет базовый useWebSocket дополнительными возможностями
 */
import { ref, computed, watch, onUnmounted } from 'vue'
import { useWebSocket, WSConnectionState, type UseWebSocketOptions } from './useWebSocket'

export interface WebSocketMetrics {
  messagesReceived: number
  messagesSent: number
  reconnectCount: number
  lastMessageTime: number
  averageLatency: number
  connectionUptime: number
  bytesReceived: number
  bytesSent: number
}

export interface ConnectionHealth {
  status: 'healthy' | 'degraded' | 'unhealthy'
  latency: number
  uptime: number
  messageRate: number // messages per second
}

/**
 * Enhanced WebSocket composable с метриками и monitoring
 */
export function useWebSocketAdvanced(options: UseWebSocketOptions = {}) {
  const baseWs = useWebSocket(options)
  
  const metrics = ref<WebSocketMetrics>({
    messagesReceived: 0,
    messagesSent: 0,
    reconnectCount: 0,
    lastMessageTime: 0,
    averageLatency: 0,
    connectionUptime: 0,
    bytesReceived: 0,
    bytesSent: 0
  })
  
  const latencyHistory: number[] = []
  const messageTimestamps: number[] = []
  let connectionStartTime = 0
  let uptimeInterval: ReturnType<typeof setInterval> | null = null
  
  // Ping-pong для измерения latency
  const pingTimestamps = new Map<string, number>()
  let pingInterval: ReturnType<typeof setInterval> | null = null
  
  /**
   * Connection health status
   */
  const connectionHealth = computed((): ConnectionHealth => {
    const latency = metrics.value.averageLatency
    const uptime = metrics.value.connectionUptime
    const messageRate = calculateMessageRate()
    
    let status: 'healthy' | 'degraded' | 'unhealthy'
    
    if (!baseWs.isConnected.value) {
      status = 'unhealthy'
    } else if (latency > 1000 || messageRate < 0.1) {
      status = 'degraded'
    } else {
      status = 'healthy'
    }
    
    return { status, latency, uptime, messageRate }
  })
  
  /**
   * Calculate message rate (messages per second)
   */
  function calculateMessageRate(): number {
    const now = Date.now()
    const recentMessages = messageTimestamps.filter(t => now - t < 10000) // last 10s
    return recentMessages.length / 10
  }
  
  /**
   * Enhanced send with metrics tracking
   */
  const enhancedSend = (data: any): boolean => {
    const messageId = crypto.randomUUID()
    const enriched = {
      ...data,
      _meta: {
        id: messageId,
        timestamp: Date.now(),
        client_version: '1.0.0'
      }
    }
    
    if (data.type === 'ping') {
      pingTimestamps.set(messageId, Date.now())
    }
    
    const sent = baseWs.send(enriched)
    if (sent) {
      metrics.value.messagesSent++
      const messageSize = new Blob([JSON.stringify(enriched)]).size
      metrics.value.bytesSent += messageSize
    }
    return sent
  }
  
  /**
   * Start periodic ping for latency measurement
   */
  function startPing() {
    if (pingInterval) return
    
    pingInterval = setInterval(() => {
      if (baseWs.isConnected.value) {
        enhancedSend({ type: 'ping' })
      }
    }, 30000) // ping every 30s
  }
  
  function stopPing() {
    if (pingInterval) {
      clearInterval(pingInterval)
      pingInterval = null
    }
  }
  
  /**
   * Handle pong messages for latency calculation
   */
  baseWs.on('pong', (data: any) => {
    const pingTime = pingTimestamps.get(data._meta?.id)
    if (pingTime) {
      const latency = Date.now() - pingTime
      latencyHistory.push(latency)
      
      // Keep only last 100 measurements
      if (latencyHistory.length > 100) {
        latencyHistory.shift()
      }
      
      metrics.value.averageLatency = 
        Math.round(latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length)
      
      pingTimestamps.delete(data._meta?.id)
    }
  })
  
  /**
   * Track all incoming messages
   */
  const originalOn = baseWs.on
  baseWs.on = ((type: any, handler: any) => {
    const wrappedHandler = (data: any) => {
      metrics.value.messagesReceived++
      metrics.value.lastMessageTime = Date.now()
      messageTimestamps.push(Date.now())
      
      // Keep only last minute of timestamps
      const cutoff = Date.now() - 60000
      while (messageTimestamps.length > 0 && messageTimestamps[0] < cutoff) {
        messageTimestamps.shift()
      }
      
      // Estimate message size
      const messageSize = new Blob([JSON.stringify(data)]).size
      metrics.value.bytesReceived += messageSize
      
      return handler(data)
    }
    return originalOn(type, wrappedHandler)
  }) as typeof originalOn
  
  /**
   * Track reconnections
   */
  watch(() => baseWs.connectionState.value, (newState, oldState) => {
    if (newState === WSConnectionState.Connecting && oldState === WSConnectionState.Reconnecting) {
      metrics.value.reconnectCount++
    }
    
    if (newState === WSConnectionState.Connected) {
      connectionStartTime = Date.now()
      startPing()
      
      // Start uptime tracking
      if (uptimeInterval) clearInterval(uptimeInterval)
      uptimeInterval = setInterval(() => {
        metrics.value.connectionUptime = Math.floor((Date.now() - connectionStartTime) / 1000)
      }, 1000)
    } else {
      stopPing()
      if (uptimeInterval) {
        clearInterval(uptimeInterval)
        uptimeInterval = null
      }
    }
  })
  
  /**
   * Reset metrics
   */
  function resetMetrics() {
    metrics.value = {
      messagesReceived: 0,
      messagesSent: 0,
      reconnectCount: 0,
      lastMessageTime: 0,
      averageLatency: 0,
      connectionUptime: 0,
      bytesReceived: 0,
      bytesSent: 0
    }
    latencyHistory.length = 0
    messageTimestamps.length = 0
    pingTimestamps.clear()
  }
  
  /**
   * Get detailed statistics
   */
  const statistics = computed(() => {
    const m = metrics.value
    return {
      totalMessages: m.messagesReceived + m.messagesSent,
      totalBytes: m.bytesReceived + m.bytesSent,
      avgMessageSize: m.messagesReceived > 0 
        ? Math.round(m.bytesReceived / m.messagesReceived) 
        : 0,
      messageRate: calculateMessageRate(),
      lastActivity: m.lastMessageTime > 0 
        ? Math.floor((Date.now() - m.lastMessageTime) / 1000) 
        : null,
      connectionQuality: getConnectionQuality()
    }
  })
  
  /**
   * Get connection quality rating (0-100)
   */
  function getConnectionQuality(): number {
    if (!baseWs.isConnected.value) return 0
    
    const latency = metrics.value.averageLatency
    const reconnects = metrics.value.reconnectCount
    const uptime = metrics.value.connectionUptime
    
    let quality = 100
    
    // Penalize high latency
    if (latency > 100) quality -= Math.min(30, (latency - 100) / 30)
    
    // Penalize reconnections
    quality -= Math.min(20, reconnects * 5)
    
    // Reward uptime
    if (uptime < 60) quality -= (60 - uptime) / 2
    
    return Math.max(0, Math.round(quality))
  }
  
  onUnmounted(() => {
    stopPing()
    if (uptimeInterval) clearInterval(uptimeInterval)
  })
  
  return {
    ...baseWs,
    send: enhancedSend,
    metrics: computed(() => metrics.value),
    connectionHealth,
    statistics,
    resetMetrics
  }
}

/**
 * Format bytes to human readable
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round(bytes / Math.pow(k, i) * 100) / 100} ${sizes[i]}`
}

/**
 * Format uptime to human readable
 */
export function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}
