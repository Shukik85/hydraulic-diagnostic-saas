/**
 * Metrics Store
 * Platform metrics and analytics state management
 */

import { defineStore } from 'pinia';
import type { PlatformMetrics, SystemHealth, RevenuePoint } from '~/types';

interface MetricPoint {
  timestamp: Date;
  value: number;
}

interface MetricsState {
  current: PlatformMetrics | null;
  history: {
    revenue: RevenuePoint[];
    tenants: MetricPoint[];
    users: MetricPoint[];
  };
  isLoading: boolean;
  error: Error | null;
  lastUpdate: Date | null;
  isSubscribed: boolean;
}

export const useMetricsStore = defineStore('metrics', {
  state: (): MetricsState => ({
    current: null,
    history: {
      revenue: [],
      tenants: [],
      users: [],
    },
    isLoading: false,
    error: null,
    lastUpdate: null,
    isSubscribed: false,
  }),

  getters: {
    /**
     * MRR trend (positive/negative percentage)
     */
    mrrTrend: (state): number => {
      return state.current?.mrrGrowthPct || 0;
    },

    /**
     * Tenant growth
     */
    tenantGrowth: (state): number => {
      if (!state.current) return 0;
      const total = state.current.totalTenants;
      const new30d = state.current.newTenants30d;
      if (total === 0) return 0;
      return (new30d / total) * 100;
    },

    /**
     * User growth
     */
    userGrowth: (state): number => {
      if (!state.current) return 0;
      const total = state.current.totalUsers;
      const new7d = state.current.newUsers7d;
      if (total === 0) return 0;
      return (new7d / total) * 100;
    },

    /**
     * System health status
     */
    healthStatus: (state): 'healthy' | 'degraded' | 'critical' | 'unknown' => {
      if (!state.current?.systemHealth) return 'unknown';
      return state.current.systemHealth.status;
    },

    /**
     * Check if metrics are fresh (< 1 minute old)
     */
    isFresh: (state): boolean => {
      if (!state.lastUpdate) return false;
      const age = Date.now() - state.lastUpdate.getTime();
      return age < 60000; // 1 minute
    },

    /**
     * Get revenue trend data for chart
     */
    revenueChartData: (state) => {
      return state.history.revenue.map((point) => ({
        x: point.date,
        y: point.value,
      }));
    },

    /**
     * Get latest revenue value
     */
    latestRevenue: (state): number => {
      if (state.history.revenue.length === 0) return 0;
      return state.history.revenue[state.history.revenue.length - 1].value;
    },
  },

  actions: {
    /**
     * Fetch current metrics from API
     */
    async fetchMetrics(): Promise<void> {
      this.isLoading = true;
      this.error = null;

      try {
        const api = useApi();
        const response = await api.get<PlatformMetrics>('/api/v1/admin/metrics');
        
        this.current = response.data;
        this.lastUpdate = new Date();
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Failed to fetch metrics');
        throw error;
      } finally {
        this.isLoading = false;
      }
    },

    /**
     * Fetch historical revenue data
     */
    async fetchRevenueHistory(days: number = 30): Promise<void> {
      try {
        const api = useApi();
        const response = await api.get<RevenuePoint[]>('/api/v1/admin/metrics/revenue', {
          days,
        });
        
        this.history.revenue = response.data;
      } catch (error) {
        console.error('Failed to fetch revenue history:', error);
      }
    },

    /**
     * Update metrics (from WebSocket)
     */
    updateMetrics(metrics: PlatformMetrics): void {
      this.current = metrics;
      this.lastUpdate = new Date();
    },

    /**
     * Update system health (from WebSocket)
     */
    updateSystemHealth(health: SystemHealth): void {
      if (this.current) {
        this.current = {
          ...this.current,
          systemHealth: health,
        };
        this.lastUpdate = new Date();
      }
    },

    /**
     * Add revenue data point to history
     */
    addRevenuePoint(point: RevenuePoint): void {
      this.history.revenue.push(point);
      
      // Keep only last 90 days
      if (this.history.revenue.length > 90) {
        this.history.revenue.shift();
      }
    },

    /**
     * Subscribe to real-time metrics updates
     */
    subscribe(): void {
      if (this.isSubscribed) return;

      const config = useRuntimeConfig();
      const wsBaseUrl = config.public.wsBase || 'ws://localhost:8000';
      
      const ws = useWebSocket(`${wsBaseUrl}/ws/admin/metrics`, {
        autoConnect: true,
        reconnect: true,
      });

      // Subscribe to metrics updates
      ws.on<PlatformMetrics>('metrics', (data) => {
        this.updateMetrics(data);
      });

      // Subscribe to system health updates
      ws.on<SystemHealth>('system_health', (data) => {
        this.updateSystemHealth(data);
      });

      // Subscribe to revenue updates
      ws.on<RevenuePoint>('revenue', (data) => {
        this.addRevenuePoint(data);
      });

      this.isSubscribed = true;
    },

    /**
     * Unsubscribe from real-time updates
     */
    unsubscribe(): void {
      // WebSocket cleanup handled by composable
      this.isSubscribed = false;
    },

    /**
     * Refresh metrics (force fetch)
     */
    async refresh(): Promise<void> {
      await this.fetchMetrics();
      await this.fetchRevenueHistory();
    },

    /**
     * Clear metrics data
     */
    clear(): void {
      this.current = null;
      this.history = {
        revenue: [],
        tenants: [],
        users: [],
      };
      this.lastUpdate = null;
      this.error = null;
    },
  },
});
