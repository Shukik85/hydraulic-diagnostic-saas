/**
 * Zod validation schemas
 */

import { z } from 'zod';

/**
 * Email validation schema
 */
export const emailSchema = z.string().email('Invalid email address');

/**
 * Password validation schema
 */
export const passwordSchema = z
  .string()
  .min(8, 'Password must be at least 8 characters')
  .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
  .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
  .regex(/[0-9]/, 'Password must contain at least one number');

/**
 * Login request schema
 */
export const loginSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, 'Password is required'),
});

/**
 * User profile schema
 */
export const userProfileSchema = z.object({
  id: z.string().uuid(),
  email: emailSchema,
  firstName: z.string().min(1, 'First name is required'),
  lastName: z.string().min(1, 'Last name is required'),
  role: z.enum(['admin', 'engineer', 'viewer']),
  organizationId: z.string().uuid().optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});

/**
 * Sensor schema
 */
export const sensorSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1, 'Sensor name is required'),
  systemId: z.string().uuid(),
  type: z.enum(['pressure', 'temperature', 'flow', 'vibration', 'position']),
  unit: z.string(),
  status: z.enum(['online', 'offline', 'error', 'calibrating']),
  lastReading: z
    .object({
      value: z.number(),
      timestamp: z.string().datetime(),
      quality: z.enum(['good', 'warning', 'bad']),
    })
    .optional(),
  metadata: z
    .object({
      minValue: z.number().optional(),
      maxValue: z.number().optional(),
      threshold: z.number().optional(),
      calibrationDate: z.string().datetime().optional(),
      location: z.string().optional(),
    })
    .optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});

/**
 * Anomaly schema
 */
export const anomalySchema = z.object({
  id: z.string().uuid(),
  sensorId: z.string().uuid(),
  systemId: z.string().uuid(),
  severity: z.enum(['critical', 'high', 'medium', 'low']),
  type: z.enum([
    'pressure_spike',
    'temperature_high',
    'flow_anomaly',
    'vibration_excessive',
    'leak_detected',
    'performance_degradation',
  ]),
  description: z.string(),
  detectedAt: z.string().datetime(),
  acknowledgedAt: z.string().datetime().optional(),
  acknowledgedBy: z.string().uuid().optional(),
  resolvedAt: z.string().datetime().optional(),
  resolvedBy: z.string().uuid().optional(),
  confidence: z.number().min(0).max(1),
  metadata: z
    .object({
      expectedValue: z.number().optional(),
      actualValue: z.number().optional(),
      deviation: z.number().optional(),
      impact: z.string().optional(),
      recommendedAction: z.string().optional(),
    })
    .optional(),
});

/**
 * Platform metrics schema
 */
export const platformMetricsSchema = z.object({
  mrr: z.number().min(0),
  mrrGrowthPct: z.number(),
  totalTenants: z.number().int().min(0),
  newTenants30d: z.number().int().min(0),
  totalUsers: z.number().int().min(0),
  newUsers7d: z.number().int().min(0),
  uptimePct: z.number().min(0).max(100),
  systemHealth: z.object({
    apiLatencyP90: z.number().min(0),
    dbLatencyP90: z.number().min(0),
    mlLatencyP90: z.number().min(0),
    status: z.enum(['healthy', 'degraded', 'critical']),
  }),
});

/**
 * Pagination schema
 */
export const paginationSchema = z.object({
  page: z.number().int().min(1),
  limit: z.number().int().min(1).max(100),
  total: z.number().int().min(0),
  totalPages: z.number().int().min(0),
  hasNext: z.boolean(),
  hasPrev: z.boolean(),
});

/**
 * Helper to parse and validate data
 */
export function validateData<T>(schema: z.ZodSchema<T>, data: unknown): T {
  return schema.parse(data);
}

/**
 * Helper to safely parse data
 */
export function safeValidateData<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; error: z.ZodError } {
  const result = schema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: result.error };
}
