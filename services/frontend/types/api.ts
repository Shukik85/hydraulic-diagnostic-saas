/**
 * TypeScript Types для Hydraulic Diagnostic Platform API
 *
 * @see https://github.com/Shukik85/hydraulic-diagnostic-saas
 * @version 1.0.0
 */

// ==================== EXISTING TYPES (PRESERVED) ====================
// !!! УДАЛЁНЫ DiagnosticResult, ChatMessage, ChatSession !!!
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

export interface DiagnosticSession {
  id: number
  system_id: number
  type: 'full' | 'pressure' | 'temperature' | 'vibration' | 'flow'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  started_at: string
  completed_at?: string
  results?: any[] // !!! DiagnosticResult удалён, оставить any[] для совместимости
  created_by: number
}

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

// Anomalies
export interface AnomaliesQueryParams {
  system_id?: number
  severity?: AnomalySeverity
  limit?: number
  offset?: number
}

export interface AnomaliesListResponse {
  items: Anomaly[]
  total: number
  limit: number
  offset: number
}

export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical'

export interface Anomaly {
  id: number
  system_id: number
  severity: AnomalySeverity
  score: number
  created_at: string
}

// System Status
export interface SystemStatus {
  id: number
  name: string
  status: 'online' | 'offline' | 'warning' | 'error'
  health: number
}

// WebSocket
export interface WSMessage {
  type: string
  payload: any
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

// AsyncState
export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: Error | null
}

export interface ErrorResponse {
  message: string
  code?: string
}

// Utility functions
export function formatAnomalyScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

export function getSeverityColor(severity: AnomalySeverity): string {
  const colors = {
    low: 'text-blue-500',
    medium: 'text-yellow-500',
    high: 'text-orange-500',
    critical: 'text-red-500'
  }
  return colors[severity] || 'text-gray-500'
}

export function getComponentStatusColor(status: string): string {
  const colors = {
    online: 'text-green-500',
    offline: 'text-gray-500',
    warning: 'text-yellow-500',
    error: 'text-red-500'
  }
  return colors[status] || 'text-gray-500'
}

// Type guard
export function isValidWSMessage(message: any): message is WSMessage {
  return typeof message === 'object' && 'type' in message && 'payload' in message
}
