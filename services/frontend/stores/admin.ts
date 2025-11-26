/**
 * Admin Store
 * Platform administration and management
 */

import { defineStore } from 'pinia';
import type { Alert, TenantUsage, AuditLog, PlanDistribution } from '~/types';

interface AdminState {
  alerts: Alert[];
  tenants: TenantUsage[];
  auditLogs: AuditLog[];
  planDistribution: PlanDistribution | null;
  isLoading: boolean;
  error: Error | null;
}

export const useAdminStore = defineStore('admin', {
  state: (): AdminState => ({
    alerts: [],
    tenants: [],
    auditLogs: [],
    planDistribution: null,
    isLoading: false,
    error: null,
  }),

  getters: {
    /**
     * Get critical alerts (sorted by severity)
     */
    criticalAlerts: (state): Alert[] => {
      return state.alerts
        .filter((a) => a.severity === 'critical' && !a.resolved)
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    },

    /**
     * Get warning alerts
     */
    warningAlerts: (state): Alert[] => {
      return state.alerts
        .filter((a) => a.severity === 'warning' && !a.resolved)
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    },

    /**
     * Count unresolved critical alerts
     */
    criticalAlertsCount: (state): number => {
      return state.alerts.filter((a) => a.severity === 'critical' && !a.resolved).length;
    },

    /**
     * Get top N tenants by usage
     */
    topTenants:
      (state) =>
      (limit: number = 10): TenantUsage[] => {
        return [...state.tenants]
          .sort((a, b) => b.sensors - a.sensors)
          .slice(0, limit);
      },

    /**
     * Get tenants by plan
     */
    tenantsByPlan:
      (state) =>
      (plan: 'starter' | 'professional' | 'enterprise'): TenantUsage[] => {
        return state.tenants.filter((t) => t.plan === plan);
      },

    /**
     * Get recent audit logs
     */
    recentAuditLogs:
      (state) =>
      (limit: number = 50): AuditLog[] => {
        return [...state.auditLogs]
          .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
          .slice(0, limit);
      },

    /**
     * Check if there are unresolved critical alerts
     */
    hasCriticalAlerts: (state): boolean => {
      return state.alerts.some((a) => a.severity === 'critical' && !a.resolved);
    },
  },

  actions: {
    /**
     * Fetch all alerts
     */
    async fetchAlerts(): Promise<void> {
      try {
        const api = useApi();
        const response = await api.get<Alert[]>('/api/v1/admin/alerts');
        
        // Convert timestamp strings to Date objects
        this.alerts = response.data.map((alert) => ({
          ...alert,
          timestamp: new Date(alert.timestamp),
          resolvedAt: alert.resolvedAt ? new Date(alert.resolvedAt) : undefined,
        }));
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
        this.error = error instanceof Error ? error : new Error('Failed to fetch alerts');
      }
    },

    /**
     * Fetch tenants usage data
     */
    async fetchTenants(): Promise<void> {
      this.isLoading = true;
      this.error = null;

      try {
        const api = useApi();
        const response = await api.get<TenantUsage[]>('/api/v1/admin/tenants');
        
        this.tenants = response.data.map((tenant) => ({
          ...tenant,
          lastActivityAt: new Date(tenant.lastActivityAt),
        }));
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Failed to fetch tenants');
        throw error;
      } finally {
        this.isLoading = false;
      }
    },

    /**
     * Fetch plan distribution
     */
    async fetchPlanDistribution(): Promise<void> {
      try {
        const api = useApi();
        const response = await api.get<PlanDistribution>('/api/v1/admin/plans/distribution');
        this.planDistribution = response.data;
      } catch (error) {
        console.error('Failed to fetch plan distribution:', error);
      }
    },

    /**
     * Fetch audit logs
     */
    async fetchAuditLogs(limit: number = 100): Promise<void> {
      try {
        const api = useApi();
        const response = await api.get<AuditLog[]>('/api/v1/admin/audit-logs', { limit });
        
        this.auditLogs = response.data.map((log) => ({
          ...log,
          timestamp: new Date(log.timestamp),
        }));
      } catch (error) {
        console.error('Failed to fetch audit logs:', error);
      }
    },

    /**
     * Dismiss alert
     */
    async dismissAlert(alertId: string): Promise<void> {
      try {
        const api = useApi();
        await api.post(`/api/v1/admin/alerts/${alertId}/dismiss`, {});
        
        // Update local state
        const alert = this.alerts.find((a) => a.id === alertId);
        if (alert) {
          alert.resolved = true;
          alert.resolvedAt = new Date();
        }
      } catch (error) {
        console.error('Failed to dismiss alert:', error);
        throw error;
      }
    },

    /**
     * Clear all alerts
     */
    async clearAlerts(): Promise<void> {
      try {
        const api = useApi();
        await api.post('/api/v1/admin/alerts/clear', {});
        
        this.alerts = [];
      } catch (error) {
        console.error('Failed to clear alerts:', error);
        throw error;
      }
    },

    /**
     * Add alert (from WebSocket)
     */
    addAlert(alert: Alert): void {
      this.alerts.unshift({
        ...alert,
        timestamp: new Date(alert.timestamp),
      });

      // Keep only last 100 alerts
      if (this.alerts.length > 100) {
        this.alerts = this.alerts.slice(0, 100);
      }
    },

    /**
     * Update tenant data
     */
    updateTenant(tenantId: string, data: Partial<TenantUsage>): void {
      const tenant = this.tenants.find((t) => t.tenantId === tenantId);
      if (tenant) {
        Object.assign(tenant, data);
      }
    },

    /**
     * Subscribe to real-time admin updates
     */
    subscribe(): void {
      const config = useRuntimeConfig();
      const wsBaseUrl = config.public.wsBase || 'ws://localhost:8000';
      
      const ws = useWebSocket(`${wsBaseUrl}/ws/admin/alerts`, {
        autoConnect: true,
        reconnect: true,
      });

      // Subscribe to new alerts
      ws.on<Alert>('alert', (data) => {
        this.addAlert(data);
      });

      // Subscribe to tenant updates
      ws.on<{ tenantId: string; data: Partial<TenantUsage> }>('tenant_update', (data) => {
        this.updateTenant(data.tenantId, data.data);
      });
    },

    /**
     * Refresh all admin data
     */
    async refresh(): Promise<void> {
      await Promise.all([
        this.fetchAlerts(),
        this.fetchTenants(),
        this.fetchPlanDistribution(),
      ]);
    },

    /**
     * Clear admin data
     */
    clear(): void {
      this.alerts = [];
      this.tenants = [];
      this.auditLogs = [];
      this.planDistribution = null;
      this.error = null;
    },
  },
});
