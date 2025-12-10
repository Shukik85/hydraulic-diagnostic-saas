/**
 * Validation Schemas
 * Zod schemas for runtime validation
 */

import { z } from 'zod';

/**
 * User validation
 */
export const userSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  firstName: z.string().min(1).max(100),
  lastName: z.string().min(1).max(100),
  role: z.enum(['admin', 'manager', 'operator', 'viewer']),
  organizationId: z.string().uuid(),
  avatar: z.string().url().optional(),
  isActive: z.boolean(),
  lastLoginAt: z.coerce.date().optional(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
});

export type UserValidation = z.infer<typeof userSchema>;

/**
 * Login validation
 */
export const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

export type LoginValidation = z.infer<typeof loginSchema>;

/**
 * Sensor validation
 */
export const sensorSchema = z.object({
  id: z.string().uuid(),
  systemId: z.string().uuid(),
  name: z.string().min(1).max(255),
  type: z.enum(['pressure', 'temperature', 'flow', 'vibration', 'position']),
  unit: z.string().min(1).max(50),
  status: z.enum(['online', 'offline', 'error', 'calibrating']),
  lastReading: z
    .object({
      value: z.number(),
      timestamp: z.coerce.date(),
      quality: z.enum(['good', 'warning', 'bad']),
    })
    .optional(),
  metadata: z.record(z.unknown()).optional(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
});

export type SensorValidation = z.infer<typeof sensorSchema>;

/**
 * Anomaly validation
 */
export const anomalySchema = z.object({
  id: z.string().uuid(),
  sensorId: z.string().uuid(),
  systemId: z.string().uuid(),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  type: z.string().min(1),
  description: z.string().min(1),
  detectedAt: z.coerce.date(),
  acknowledgedAt: z.coerce.date().optional(),
  acknowledgedBy: z.string().uuid().optional(),
  resolvedAt: z.coerce.date().optional(),
  resolvedBy: z.string().uuid().optional(),
  confidence: z.number().min(0).max(1),
  metadata: z.record(z.unknown()).optional(),
});

export type AnomalyValidation = z.infer<typeof anomalySchema>;

/**
 * API Response validation
 */
export const apiResponseSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema,
    message: z.string().optional(),
    timestamp: z.string(),
  });

/**
 * Paginated Response validation
 */
export const paginatedResponseSchema = <T extends z.ZodTypeAny>(itemSchema: T) =>
  z.object({
    items: z.array(itemSchema),
    pagination: z.object({
      page: z.number().int().positive(),
      limit: z.number().int().positive(),
      total: z.number().int().nonnegative(),
      totalPages: z.number().int().nonnegative(),
      hasNext: z.boolean(),
      hasPrev: z.boolean(),
    }),
  });

/**
 * Platform Metrics validation
 */
export const platformMetricsSchema = z.object({
  mrr: z.number().nonnegative(),
  mrrGrowthPct: z.number(),
  totalTenants: z.number().int().nonnegative(),
  newTenants30d: z.number().int().nonnegative(),
  totalUsers: z.number().int().nonnegative(),
  newUsers7d: z.number().int().nonnegative(),
  uptimePct: z.number().min(0).max(100),
  systemHealth: z.object({
    apiLatencyP90: z.number().nonnegative(),
    dbLatencyP90: z.number().nonnegative(),
    mlLatencyP90: z.number().nonnegative(),
    errorRate: z.number().min(0).max(1),
    requestsPerSecond: z.number().nonnegative(),
    status: z.enum(['healthy', 'degraded', 'critical']),
  }),
});

export type PlatformMetricsValidation = z.infer<typeof platformMetricsSchema>;
