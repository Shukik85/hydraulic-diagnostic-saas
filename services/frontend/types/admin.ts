/**
 * Admin dashboard types
 */

import type { AnomalySeverity } from './models';

/**
 * Platform metrics for admin dashboard
 */
export interface PlatformMetrics {
  mrr: number;
  mrrGrowthPct: number;
  totalTenants: number;
  newTenants30d: number;
  totalUsers: number;
  newUsers7d: number;
  uptimePct: number;
  systemHealth: SystemHealth;
}

/**
 * System health metrics
 */
export interface SystemHealth {
  apiLatencyP90: number;
  dbLatencyP90: number;
  mlLatencyP90: number;
  status: HealthStatus;
}

/**
 * Health status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'critical';

/**
 * Revenue data point
 */
export interface RevenuePoint {
  date: string;
  value: number;
}

/**
 * Alert
 */
export interface Alert {
  id: string;
  severity: AnomalySeverity;
  message: string;
  timestamp: Date;
  source: string;
  resolved: boolean;
  resolvedAt?: Date;
  resolvedBy?: string;
}

/**
 * Tenant usage statistics
 */
export interface TenantUsage {
  tenantId: string;
  tenantName: string;
  sensors: number;
  apiCalls: number;
  storage: number;
  users: number;
  plan: 'starter' | 'professional' | 'enterprise';
}

/**
 * Tier distribution
 */
export interface TierDistribution {
  starter: number;
  professional: number;
  enterprise: number;
}

/**
 * Audit log entry
 */
export interface AuditLogEntry {
  id: string;
  userId: string;
  userName: string;
  action: string;
  resource: string;
  resourceId?: string;
  timestamp: Date;
  ipAddress?: string;
  userAgent?: string;
  metadata?: Record<string, unknown>;
}
