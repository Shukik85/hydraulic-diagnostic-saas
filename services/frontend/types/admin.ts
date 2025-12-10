/**
 * Admin Types
 * Platform administration and metrics
 */

/**
 * Platform-wide metrics
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
  errorRate: number;
  requestsPerSecond: number;
  status: 'healthy' | 'degraded' | 'critical';
}

/**
 * Platform alert
 */
export interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  source: string;
  timestamp: Date;
  resolved?: boolean;
  resolvedAt?: Date;
  resolvedBy?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Tenant usage statistics
 */
export interface TenantUsage {
  tenantId: string;
  tenantName: string;
  plan: 'starter' | 'professional' | 'enterprise';
  sensors: number;
  apiCalls: number;
  storage: number; // bytes
  users: number;
  lastActivityAt: Date;
}

/**
 * Revenue data point
 */
export interface RevenuePoint {
  date: string; // ISO date string
  value: number;
  tenants: number;
}

/**
 * Plan distribution
 */
export interface PlanDistribution {
  starter: number;
  professional: number;
  enterprise: number;
}

/**
 * Audit log entry
 */
export interface AuditLog {
  id: string;
  userId: string;
  userName: string;
  action: string;
  resource: string;
  resourceId?: string;
  changes?: Record<string, unknown>;
  ipAddress: string;
  userAgent: string;
  timestamp: Date;
}
