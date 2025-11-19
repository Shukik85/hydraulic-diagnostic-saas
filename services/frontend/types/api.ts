/**
 * TypeScript Types для Hydraulic Diagnostic Platform API
 *
 * @see https://github.com/Shukik85/hydraulic-diagnostic-saas
 * @version 1.0.1 - Fixed missing properties
 */

// ==================== BASE TYPES ====================

export interface User {
  id: number
  email: string
  username?: string
  name?: string
  first_name?: string
  last_name?: string
  avatar?: string
  is_active: boolean
  is_staff?: boolean
  date_joined: string
  last_login?: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterData {
  email: string
  password: string
  password_confirm: string
  first_name?: string
  last_name?: string
  username?: string
}

export interface AuthTokens {
  access: string
  refresh: string
}

export interface ApiResponse<T = any> {
  data: T
  message?: string
  error?: string
}

// ==================== SYSTEM TYPES ====================

export interface HydraulicSystem {
  id: number
  name: string
  type: 'industrial' | 'mobile' | 'marine' | 'construction' | 'mining' | 'agricultural'
  status: 'active' | 'maintenance' | 'inactive'
  description?: string
  location?: string
  pressure: number
  temperature: number
  flow_rate?: number
  vibration?: number
  health_score: number
  last_update: string
  created_at: string
  updated_at: string
}

export interface SystemStatus {
  id: number
  name: string
  status: 'online' | 'offline' | 'warning' | 'error'
  health: number
  health_score?: number  // Alias for compatibility
  component_statuses?: ComponentStatus[]
}

export interface ComponentStatus {
  component_id: number
  name: string
  status: 'online' | 'offline' | 'warning' | 'error'
  value?: number
}

// ==================== ANOMALIES ====================

export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical'

export interface Anomaly {
  id: number
  prediction_id?: number  // Alias for id
  system_id: number
  severity: AnomalySeverity
  score: number
  anomaly_score?: number  // Alias for score
  created_at: string
  description?: string
}

export interface AnomaliesQueryParams {
  system_id?: number
  severity?: AnomalySeverity
  limit?: number
  offset?: number
  page?: number
  per_page?: number
  start_date?: string
  end_date?: string
}

export interface AnomaliesListResponse {
  items: Anomaly[]
  total: number
  limit: number
  offset: number
  pagination?: {
    page: number
    per_page: number
    total: number
    pages: number
  }
}

// ==================== DIAGNOSTICS ====================

export interface DiagnosticSession {
  id: number
  system_id: number
  type: 'full' | 'pressure' | 'temperature' | 'vibration' | 'flow'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  started_at: string
  completed_at?: string
  results?: any[]
  created_by: number
}

// ==================== WEBSOCKET TYPES ====================

export interface WSMessage {
  type: string
  payload: any
  data?: any  // For backward compatibility
}

export interface WSNewSensorReading {
  sensor_id: number
  value: number
  timestamp: string
}

export interface WSNewAnomaly {
  anomaly_id: number
  severity: AnomalySeverity
  message: string
}

export interface WSSystemStatusUpdate {
  system_id: number
  status: string
}

// Type guard
export function isValidWSMessage(message: any): message is WSMessage {
  return typeof message === 'object' && 'type' in message && ('payload' in message || 'data' in message)
}

// ==================== ASYNC STATE ====================

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: ErrorResponse | null
}

export interface ErrorResponse {
  message: string
  code?: string
  name?: string  // For Error compatibility
  timestamp?: string
  request_id?: string
  error?: {
    message: string
    code?: string
    timestamp?: string
    request_id?: string
  }
}

// ==================== UI TYPES ====================

export interface ApiError {
  message: string
  status: number
  data?: any
}

export interface TableColumn {
  key: string
  label: string
  sortable?: boolean
  width?: string
  align?: 'left' | 'center' | 'right'
}

export interface UiPasswordStrength {
  score: number
  label: 'weak' | 'fair' | 'good' | 'strong'
  color: 'red' | 'yellow' | 'green' | 'blue'
}

export interface PasswordStrength {
  score: number
  feedback: {
    warning: string
    suggestions: string[]
  }
  crack_times_seconds: {
    online_throttling_100_per_hour: number
    online_no_throttling_10_per_second: number
    offline_slow_hashing_1e4_per_second: number
    offline_fast_hashing_1e10_per_second: number
  }
  crack_times_display: {
    online_throttling_100_per_hour: string
    online_no_throttling_10_per_second: string
    offline_slow_hashing_1e4_per_second: string
    offline_fast_hashing_1e10_per_second: string
  }
}

// ==================== UTILITY FUNCTIONS ====================

export function formatAnomalyScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

export function getSeverityColor(severity: AnomalySeverity): string {
  const colors: Record<AnomalySeverity, string> = {
    low: 'text-blue-500',
    medium: 'text-yellow-500',
    high: 'text-orange-500',
    critical: 'text-red-500'
  }
  return colors[severity] || 'text-gray-500'
}

export function getComponentStatusColor(status: string): string {
  const colors: Record<string, string> = {
    online: 'text-green-500',
    offline: 'text-gray-500',
    warning: 'text-yellow-500',
    error: 'text-red-500'
  }
  return colors[status as keyof typeof colors] || 'text-gray-500'
}
