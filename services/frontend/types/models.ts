/**
 * Domain Models
 * Core business entities
 */

/**
 * User roles
 */
export type UserRole = 'admin' | 'manager' | 'operator' | 'viewer';

/**
 * User model
 */
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  organizationId: string;
  avatar?: string;
  isActive: boolean;
  lastLoginAt?: Date;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Organization/Tenant model
 */
export interface Organization {
  id: string;
  name: string;
  slug: string;
  plan: 'starter' | 'professional' | 'enterprise';
  logo?: string;
  settings: OrganizationSettings;
  createdAt: Date;
  updatedAt: Date;
}

export interface OrganizationSettings {
  maxSensors: number;
  maxUsers: number;
  features: string[];
  customDomain?: string;
}

/**
 * Hydraulic system
 */
export interface HydraulicSystem {
  id: string;
  organizationId: string;
  name: string;
  description?: string;
  location?: string;
  status: 'active' | 'inactive' | 'maintenance';
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Sensor types
 */
export type SensorType = 'pressure' | 'temperature' | 'flow' | 'vibration' | 'position';

/**
 * Sensor model
 */
export interface Sensor {
  id: string;
  systemId: string;
  name: string;
  type: SensorType;
  unit: string;
  status: 'online' | 'offline' | 'error' | 'calibrating';
  lastReading?: SensorReading;
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Sensor reading
 */
export interface SensorReading {
  value: number;
  timestamp: Date;
  quality: 'good' | 'warning' | 'bad';
}

/**
 * Anomaly severity
 */
export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical';

/**
 * Anomaly model
 */
export interface Anomaly {
  id: string;
  sensorId: string;
  systemId: string;
  severity: AnomalySeverity;
  type: string;
  description: string;
  detectedAt: Date;
  acknowledgedAt?: Date;
  acknowledgedBy?: string;
  resolvedAt?: Date;
  resolvedBy?: string;
  confidence: number;
  metadata?: Record<string, unknown>;
}

/**
 * ML Model
 */
export interface MLModel {
  id: string;
  name: string;
  version: string;
  type: 'gnn' | 'lstm' | 'transformer';
  accuracy: number;
  status: 'training' | 'deployed' | 'deprecated';
  trainedAt?: Date;
  deployedAt?: Date;
  metrics?: Record<string, number>;
}
