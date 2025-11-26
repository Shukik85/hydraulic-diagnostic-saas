/**
 * Metrics Composable
 * Platform metrics with real-time WebSocket updates
 */

import type { PlatformMetrics, SystemHealth } from '~/types';

interface MetricsState {
  data: PlatformMetrics | null;
  isLoading: boolean;
  error: Error | null;
  lastUpdate: Date | null;
}

const CACHE_TTL = 60000; // 1 minute
const metricsCache = ref<MetricsState>({
  data: null,
  isLoading: false,
  error: null,
  lastUpdate: null,
});

export function useMetrics() {
  const api = useApi();
  const config = useRuntimeConfig();
  const wsBaseUrl = config.public.wsBase || 'ws://localhost:8000';

  let ws: ReturnType<typeof useWebSocket> | null = null;

  /**
   * Check if cache is valid
   */
  function isCacheValid(): boolean {
    if (!metricsCache.value.data || !metricsCache.value.lastUpdate) {
      return false;
    }

    const age = Date.now() - metricsCache.value.lastUpdate.getTime();
    return age < CACHE_TTL;
  }

  /**
   * Fetch metrics from API
   */
  async function fetchMetrics(): Promise<void> {
    // Return cached data if valid
    if (isCacheValid()) {
      return;
    }

    metricsCache.value.isLoading = true;
    metricsCache.value.error = null;

    try {
      const response = await api.get<PlatformMetrics>('/api/v1/admin/metrics');
      metricsCache.value.data = response.data;
      metricsCache.value.lastUpdate = new Date();
    } catch (error) {
      metricsCache.value.error = error instanceof Error ? error : new Error('Failed to fetch metrics');
      throw error;
    } finally {
      metricsCache.value.isLoading = false;
    }
  }

  /**
   * Subscribe to real-time metrics updates
   */
  function subscribe(): void {
    if (ws) return; // Already subscribed

    ws = useWebSocket(`${wsBaseUrl}/ws/admin/metrics`, {
      autoConnect: true,
      reconnect: true,
    });

    // Handle metrics updates
    ws.on<PlatformMetrics>('metrics', (data) => {
      metricsCache.value.data = data;
      metricsCache.value.lastUpdate = new Date();
    });

    // Handle system health updates
    ws.on<SystemHealth>('system_health', (data) => {
      if (metricsCache.value.data) {
        metricsCache.value.data = {
          ...metricsCache.value.data,
          systemHealth: data,
        };
        metricsCache.value.lastUpdate = new Date();
      }
    });
  }

  /**
   * Unsubscribe from real-time updates
   */
  function unsubscribe(): void {
    if (ws) {
      ws.disconnect();
      ws = null;
    }
  }

  /**
   * Refresh metrics (force fetch)
   */
  async function refresh(): Promise<void> {
    metricsCache.value.lastUpdate = null; // Invalidate cache
    await fetchMetrics();
  }

  // Computed values for easier access
  const metrics = computed(() => metricsCache.value.data);
  const isLoading = computed(() => metricsCache.value.isLoading);
  const error = computed(() => metricsCache.value.error);
  const lastUpdate = computed(() => metricsCache.value.lastUpdate);

  // Specific metric getters
  const mrr = computed(() => metrics.value?.mrr || 0);
  const mrrGrowth = computed(() => metrics.value?.mrrGrowthPct || 0);
  const totalTenants = computed(() => metrics.value?.totalTenants || 0);
  const newTenants = computed(() => metrics.value?.newTenants30d || 0);
  const totalUsers = computed(() => metrics.value?.totalUsers || 0);
  const newUsers = computed(() => metrics.value?.newUsers7d || 0);
  const uptime = computed(() => metrics.value?.uptimePct || 0);
  const systemHealth = computed(() => metrics.value?.systemHealth);

  // Health status computed
  const healthStatus = computed(() => {
    if (!systemHealth.value) return 'unknown';
    return systemHealth.value.status;
  });

  // Cleanup on unmount
  onUnmounted(() => {
    unsubscribe();
  });

  return {
    // State
    metrics,
    isLoading,
    error,
    lastUpdate,

    // Specific metrics
    mrr,
    mrrGrowth,
    totalTenants,
    newTenants,
    totalUsers,
    newUsers,
    uptime,
    systemHealth,
    healthStatus,

    // Methods
    fetchMetrics,
    subscribe,
    unsubscribe,
    refresh,
  };
}
