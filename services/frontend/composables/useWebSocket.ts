/**
 * WebSocket Composable
 * Real-time communication with auto-reconnect
 */

import type { WebSocketMessage } from '~/types';

interface WebSocketOptions {
  autoConnect?: boolean;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  reconnectAttempts: number;
}

type MessageHandler<T = unknown> = (data: T) => void;

const DEFAULT_OPTIONS: Required<WebSocketOptions> = {
  autoConnect: true,
  reconnect: true,
  reconnectInterval: 1000,
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000,
};

export function useWebSocket(url: string, options: WebSocketOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  const state = ref<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectAttempts: 0,
  });

  let ws: WebSocket | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  const messageHandlers = new Map<string, Set<MessageHandler>>();

  /**
   * Calculate reconnect delay with exponential backoff
   */
  function getReconnectDelay(): number {
    const baseDelay = opts.reconnectInterval;
    const attempt = state.value.reconnectAttempts;
    const delay = baseDelay * Math.pow(2, attempt);
    return Math.min(delay, 30000); // Max 30 seconds
  }

  /**
   * Start heartbeat to keep connection alive
   */
  function startHeartbeat(): void {
    if (heartbeatInterval) clearInterval(heartbeatInterval);

    heartbeatInterval = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        send({ type: 'ping', data: null, timestamp: new Date().toISOString() });
      }
    }, opts.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  function stopHeartbeat(): void {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
  }

  /**
   * Connect to WebSocket
   */
  function connect(): void {
    if (state.value.isConnected || state.value.isConnecting) return;

    state.value.isConnecting = true;
    state.value.error = null;

    try {
      ws = new WebSocket(url);

      ws.onopen = () => {
        state.value.isConnected = true;
        state.value.isConnecting = false;
        state.value.reconnectAttempts = 0;
        startHeartbeat();
        console.log('[WebSocket] Connected to', url);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
        }
      };

      ws.onerror = (event) => {
        state.value.error = new Error('WebSocket error');
        console.error('[WebSocket] Error:', event);
      };

      ws.onclose = () => {
        state.value.isConnected = false;
        state.value.isConnecting = false;
        stopHeartbeat();
        console.log('[WebSocket] Disconnected');

        // Attempt reconnect if enabled
        if (opts.reconnect && state.value.reconnectAttempts < opts.maxReconnectAttempts) {
          const delay = getReconnectDelay();
          console.log(`[WebSocket] Reconnecting in ${delay}ms...`);
          
          state.value.reconnectAttempts++;
          reconnectTimeout = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (error) {
      state.value.error = error instanceof Error ? error : new Error('Connection failed');
      state.value.isConnecting = false;
    }
  }

  /**
   * Disconnect from WebSocket
   */
  function disconnect(): void {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }

    stopHeartbeat();

    if (ws) {
      ws.close();
      ws = null;
    }

    state.value.isConnected = false;
    state.value.isConnecting = false;
    state.value.reconnectAttempts = 0;
  }

  /**
   * Send message through WebSocket
   */
  function send<T = unknown>(message: WebSocketMessage<T>): void {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send message: not connected');
      return;
    }

    try {
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('[WebSocket] Failed to send message:', error);
    }
  }

  /**
   * Handle incoming message
   */
  function handleMessage<T = unknown>(message: WebSocketMessage<T>): void {
    const handlers = messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message.data));
    }
  }

  /**
   * Subscribe to message type
   */
  function on<T = unknown>(type: string, handler: MessageHandler<T>): () => void {
    if (!messageHandlers.has(type)) {
      messageHandlers.set(type, new Set());
    }

    const handlers = messageHandlers.get(type)!;
    handlers.add(handler as MessageHandler);

    // Return unsubscribe function
    return () => {
      handlers.delete(handler as MessageHandler);
      if (handlers.size === 0) {
        messageHandlers.delete(type);
      }
    };
  }

  /**
   * Subscribe to multiple message types
   */
  function onMany(subscriptions: Record<string, MessageHandler>): () => void {
    const unsubscribers = Object.entries(subscriptions).map(([type, handler]) =>
      on(type, handler)
    );

    // Return combined unsubscribe function
    return () => {
      unsubscribers.forEach((unsub) => unsub());
    };
  }

  // Auto-connect if enabled
  if (opts.autoConnect && typeof window !== 'undefined') {
    connect();
  }

  // Cleanup on unmount
  onUnmounted(() => {
    disconnect();
  });

  return {
    // State
    isConnected: computed(() => state.value.isConnected),
    isConnecting: computed(() => state.value.isConnecting),
    error: computed(() => state.value.error),
    reconnectAttempts: computed(() => state.value.reconnectAttempts),

    // Methods
    connect,
    disconnect,
    send,
    on,
    onMany,
  };
}
