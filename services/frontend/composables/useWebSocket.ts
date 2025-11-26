/**
 * WebSocket composable for real-time updates
 */

export interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export interface UseWebSocketReturn {
  isConnected: Ref<boolean>;
  data: Ref<unknown>;
  connect: () => void;
  disconnect: () => void;
  send: (data: unknown) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
}

/**
 * WebSocket composable
 */
export const useWebSocket = (
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn => {
  const config = useRuntimeConfig();
  const { accessToken } = useAuth();

  const {
    autoConnect = true,
    reconnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 1000,
  } = options;

  const ws = ref<WebSocket | null>(null);
  const isConnected = ref(false);
  const data = ref<unknown>(null);
  const reconnectCount = ref(0);

  /**
   * Build WebSocket URL with auth token
   */
  const buildUrl = (): string => {
    const wsUrl = `${config.public.wsBase}${url}`;
    if (accessToken.value) {
      return `${wsUrl}?token=${accessToken.value}`;
    }
    return wsUrl;
  };

  /**
   * Connect to WebSocket
   */
  const connect = (): void => {
    if (ws.value) {
      return;
    }

    const socket = new WebSocket(buildUrl());

    socket.onopen = () => {
      isConnected.value = true;
      reconnectCount.value = 0;
      console.log('[WebSocket] Connected:', url);
    };

    socket.onmessage = (event: MessageEvent) => {
      try {
        data.value = JSON.parse(event.data);
      } catch {
        data.value = event.data;
      }
    };

    socket.onerror = (error: Event) => {
      console.error('[WebSocket] Error:', error);
    };

    socket.onclose = () => {
      isConnected.value = false;
      ws.value = null;
      console.log('[WebSocket] Disconnected:', url);

      // Auto-reconnect
      if (reconnect && reconnectCount.value < reconnectAttempts) {
        reconnectCount.value++;
        const delay = reconnectDelay * reconnectCount.value;
        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectCount.value})`);
        setTimeout(connect, delay);
      }
    };

    ws.value = socket;
  };

  /**
   * Disconnect from WebSocket
   */
  const disconnect = (): void => {
    if (ws.value) {
      ws.value.close();
      ws.value = null;
      isConnected.value = false;
    }
  };

  /**
   * Send data through WebSocket
   */
  const send = (payload: unknown): void => {
    if (ws.value && isConnected.value) {
      ws.value.send(JSON.stringify(payload));
    } else {
      console.warn('[WebSocket] Cannot send data: not connected');
    }
  };

  /**
   * Subscribe to channel
   */
  const subscribe = (channel: string): void => {
    send({
      type: 'subscribe',
      channel,
    });
  };

  /**
   * Unsubscribe from channel
   */
  const unsubscribe = (channel: string): void => {
    send({
      type: 'unsubscribe',
      channel,
    });
  };

  // Auto-connect on mount
  if (autoConnect) {
    onMounted(() => {
      connect();
    });
  }

  // Disconnect on unmount
  onUnmounted(() => {
    disconnect();
  });

  return {
    isConnected,
    data,
    connect,
    disconnect,
    send,
    subscribe,
    unsubscribe,
  };
};
