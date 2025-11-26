/**
 * Platform metrics store with real-time WebSocket updates
 */

import { defineStore } from 'pinia';
import type { PlatformMetrics, RevenuePoint, TierDistribution } from '~/types';

export const useMetricsStore = defineStore('metrics', () => {
  // State
  const metrics = ref<PlatformMetrics | null>(null);
  const revenueHistory = ref<RevenuePoint[]>([]);
  const tierDistribution = ref<TierDistribution | null>(null);
  const lastUpdate = ref<Date | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const isSubscribed = ref(false);

  // WebSocket connection
  let wsConnection: ReturnType<typeof useWebSocket> | null = null;

  // Getters
  const mrrTrend = computed(() => {
    if (!metrics.value) {
      return 0;
    }
    return metrics.value.mrrGrowthPct;
  });

  const tenantGrowth = computed(() => {
    if (!metrics.value) {
      return 0;
    }
    return metrics.value.newTenants30d;
  });

  const healthStatus = computed(() => {
    if (!metrics.value) {
      return 'unknown';
    }
    return metrics.value.systemHealth.status;
  });

  const isHealthy = computed(() => healthStatus.value === 'healthy');

  // Actions
  const fetchMetrics = async (): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      const api = useApi();
      const data = await api.get<PlatformMetrics>('/admin/metrics');
      metrics.value = data;
      lastUpdate.value = new Date();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch metrics';
      throw err;
    } finally {
      isLoading.value = false;
    }
  };

  const fetchRevenueHistory = async (days = 30): Promise<void> => {
    try {
      const api = useApi();
      const data = await api.get<RevenuePoint[]>(`/admin/metrics/revenue?days=${days}`);
      revenueHistory.value = data;
    } catch (err) {
      console.error('Failed to fetch revenue history:', err);
    }
  };

  const fetchTierDistribution = async (): Promise<void> => {
    try {
      const api = useApi();
      const data = await api.get<TierDistribution>('/admin/metrics/tiers');
      tierDistribution.value = data;
    } catch (err) {
      console.error('Failed to fetch tier distribution:', err);
    }
  };

  const subscribeToMetrics = (): void => {
    if (isSubscribed.value || wsConnection) {
      return;
    }

    wsConnection = useWebSocket('/admin/metrics', {
      autoConnect: true,
      reconnect: true,
    });

    // Listen for metrics updates
    watch(
      () => wsConnection?.data.value,
      (newData) => {
        if (newData && typeof newData === 'object' && 'mrr' in newData) {
          metrics.value = newData as PlatformMetrics;
          lastUpdate.value = new Date();
        }
      }
    );

    isSubscribed.value = true;
  };

  const unsubscribeFromMetrics = (): void => {
    if (wsConnection) {
      wsConnection.disconnect();
      wsConnection = null;
    }
    isSubscribed.value = false;
  };

  const refresh = async (): Promise<void> => {
    await Promise.all([fetchMetrics(), fetchRevenueHistory(), fetchTierDistribution()]);
  };

  return {
    // State
    metrics,
    revenueHistory,
    tierDistribution,
    lastUpdate,
    isLoading,
    error,
    isSubscribed,

    // Getters
    mrrTrend,
    tenantGrowth,
    healthStatus,
    isHealthy,

    // Actions
    fetchMetrics,
    fetchRevenueHistory,
    fetchTierDistribution,
    subscribeToMetrics,
    unsubscribeFromMetrics,
    refresh,
  };
});
