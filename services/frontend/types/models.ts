/**
 * Domain models
 */

import type { UserRole } from './api';

/**
 * User model
 */
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  organizationId?: string;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Hydraulic system model
 */
export interface HydraulicSystem {
  id: string;
  name: string;
  description?: string;
  organizationId: string;
  metadata?: SystemMetadata;
  status: SystemStatus;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * System metadata
 */
export interface SystemMetadata {
  manufacturer?: string;
  model?: string;
  serialNumber?: string;
  installationDate?: Date;
  location?: string;
  notes?: string;
}

/**
 * System status
 */
export type SystemStatus = 'active' | 'maintenance' | 'inactive' | 'error';

/**
 * Sensor model
 */
export interface Sensor {
  id: string;
  name: string;
  systemId: string;
  type: SensorType;
  unit: string;
  status: SensorStatus;
  lastReading?: SensorReading;
  metadata?: SensorMetadata;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Sensor type
 */
export type SensorType = 'pressure' | 'temperature' | 'flow' | 'vibration' | 'position';

/**
 * Sensor status
 */
export type SensorStatus = 'online' | 'offline' | 'error' | 'calibrating';

/**
 * Sensor reading
 */
export interface SensorReading {
  value: number;
  timestamp: Date;
  quality: 'good' | 'warning' | 'bad';
}

/**
 * Sensor metadata
 */
export interface SensorMetadata {
  minValue?: number;
  maxValue?: number;
  threshold?: number;
  calibrationDate?: Date;
  location?: string;
}

/**
 * Anomaly model
 */
export interface Anomaly {
  id: string;
  sensorId: string;
  systemId: string;
  severity: AnomalySeverity;
  type: AnomalyType;
  description: string;
  detectedAt: Date;
  acknowledgedAt?: Date;
  acknowledgedBy?: string;
  resolvedAt?: Date;
  resolvedBy?: string;
  confidence: number;
  metadata?: AnomalyMetadata;
}

/**
 * Anomaly severity
 */
export type AnomalySeverity = 'critical' | 'high' | 'medium' | 'low';

/**
 * Anomaly type
 */
export type AnomalyType =
  | 'pressure_spike'
  | 'temperature_high'
  | 'flow_anomaly'
  | 'vibration_excessive'
  | 'leak_detected'
  | 'performance_degradation';

/**
 * Anomaly metadata
 */
export interface AnomalyMetadata {
  expectedValue?: number;
  actualValue?: number;
  deviation?: number;
  impact?: string;
  recommendedAction?: string;
}

/**
 * ML Model
 */
export interface MLModel {
  id: string;
  name: string;
  version: string;
  type: MLModelType;
  status: MLModelStatus;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  trainedAt?: Date;
  deployedAt?: Date;
  metadata?: MLModelMetadata;
}

/**
 * ML Model type
 */
export type MLModelType = 'gnn' | 'lstm' | 'transformer' | 'ensemble';

/**
 * ML Model status
 */
export type MLModelStatus = 'training' | 'deployed' | 'archived' | 'failed';

/**
 * ML Model metadata
 */
export interface MLModelMetadata {
  framework?: string;
  hyperparameters?: Record<string, unknown>;
  datasetSize?: number;
  trainingDuration?: number;
  notes?: string;
}
