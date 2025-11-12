/**
 * TypeScript Types for Hydraulic Diagnostic Platform API
 * 
 * @see https://github.com/Shukik85/hydraulic-diagnostic-saas
 * @version 1.0.0
 */

// ==================== EXISTING TYPES (PRESERVED) ====================

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
  results?: DiagnosticResult[]
  created_by: number
}

export interface DiagnosticResult {
  id: number
  session_id: number
  metric_type: string
  value: number
  threshold_min?: number
  threshold_max?: number
  status: 'normal' | 'warning' | 'critical'
  recommendations?: string[]
  timestamp: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: {
    title: string
    url: string
  }[]
}

export interface ChatSession {
  id: number
  title: string
  description: string
  lastMessage: string
  timestamp: string
  messages: ChatMessage[]
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

// ==================== NEW OPENAPI v3.1 TYPES ====================

// -------------------- ENUMS --------------------

/**
 * Sensor measurement units
 */
export enum SensorUnit {
  Bar = 'bar',
  Celsius = 'celsius',
  RPM = 'rpm',
  LPM = 'lpm'
}

/**
 * ML model preferences
 */
export enum ModelPreference {
  CatBoost = 'catboost',
  XGBoost = 'xgboost',
  RandomForest = 'random_forest',
  Adaptive = 'adaptive',
  Ensemble = 'ensemble'
}

/**
 * Anomaly severity levels
 */
export enum AnomalySeverity {
  Normal = 'normal',
  Warning = 'warning',
  Critical = 'critical'
}

/**
 * System component types
 */
export enum ComponentType {
  Pump = 'pump',
  Valve = 'valve',
  Accumulator = 'accumulator',
  Cooler = 'cooler'
}

/**
 * Component status
 */
export enum ComponentStatus {
  Normal = 'normal',
  Degraded = 'degraded',
  Failed = 'failed'
}

/**
 * Job processing status
 */
export enum JobStatus {
  Queued = 'queued',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed'
}

/**
 * API error codes
 */
export enum ErrorCode {
  ValidationError = 'VALIDATION_ERROR',
  AuthenticationError = 'AUTHENTICATION_ERROR',
  RateLimitExceeded = 'RATE_LIMIT_EXCEEDED',
  InternalError = 'INTERNAL_ERROR',
  MLServiceUnavailable = 'ML_SERVICE_UNAVAILABLE',
  NotFound = 'NOT_FOUND',
  Forbidden = 'FORBIDDEN',
  NetworkError = 'NETWORK_ERROR' // Added for network failures
}

// -------------------- REQUEST SCHEMAS --------------------

/**
 * Single sensor reading
 */
export interface SensorReading {
  /** Sensor UUID */
  sensor_id: string
  /** ISO 8601 timestamp */
  timestamp: string
  /** Numeric reading value */
  value: number
  /** Measurement unit */
  unit: SensorUnit
  /** Data quality (0-100%) */
  quality?: number
}

/**
 * Bulk sensor data ingestion
 */
export interface SensorBulkIngest {
  /** System UUID */
  system_id: string
  /** Array of readings (1-10000) */
  readings: SensorReading[]
}

/**
 * ML prediction request
 */
export interface MLPredictionRequest {
  /** System UUID */
  system_id: string
  /** Sensor data for analysis */
  sensor_data: SensorReading[]
  /** Preferred model (default: ensemble) */
  model_preference?: ModelPreference
}

// -------------------- RESPONSE SCHEMAS --------------------

/**
 * Bulk ingestion response
 */
export interface BulkIngestResponse {
  /** Processing job UUID */
  job_id: string
  /** Processing status */
  status: JobStatus
}

/**
 * Single model prediction details
 */
export interface ModelPrediction {
  /** Model name */
  model: ModelPreference
  /** Anomaly score (0-1) */
  score: number
  /** Model confidence (0-1) */
  confidence: number
}

/**
 * ML inference metadata
 */
export interface PredictionMetadata {
  /** Inference time in milliseconds */
  inference_time_ms: number
  /** Model version */
  model_version: string
  /** Was cache used */
  cache_hit: boolean
}

/**
 * ML prediction response
 */
export interface MLPredictionResponse {
  /** Prediction UUID */
  prediction_id: string
  /** System UUID */
  system_id: string
  /** ISO 8601 timestamp */
  timestamp: string
  /** Overall anomaly score (0-1) */
  anomaly_score: number
  /** Severity level */
  severity: AnomalySeverity
  /** Predictions from different models */
  predictions: ModelPrediction[]
  /** Inference metadata */
  metadata: PredictionMetadata
}

/**
 * Component status details
 */
export interface ComponentStatusDetail {
  /** Component ID */
  component_id: string
  /** Component type */
  type: ComponentType
  /** Current status */
  status: ComponentStatus
  /** Last anomaly timestamp */
  last_anomaly: string | null
}

/**
 * System overall status
 */
export interface SystemStatus {
  /** System UUID */
  system_id: string
  /** Overall health score (0-100) */
  health_score: number
  /** Component statuses */
  component_statuses: ComponentStatusDetail[]
  /** Last update timestamp */
  last_updated: string
}

/**
 * Pagination info
 */
export interface Pagination {
  /** Current page */
  page: number
  /** Items per page */
  per_page: number
  /** Total items */
  total: number
  /** Total pages */
  pages: number
}

/**
 * Paginated anomalies list
 */
export interface AnomaliesListResponse {
  /** Anomaly items */
  items: MLPredictionResponse[]
  /** Pagination info */
  pagination: Pagination
}

// -------------------- ERROR HANDLING --------------------

/**
 * API error details
 */
export interface APIErrorDetail {
  /** Error code */
  code: ErrorCode
  /** Human-readable message */
  message: string
  /** Additional details */
  details?: Record<string, any>
  /** ISO 8601 timestamp */
  timestamp: string
  /** Request UUID for tracing */
  request_id: string
}

/**
 * Standard error response format
 */
export interface ErrorResponse {
  error: APIErrorDetail
}

// -------------------- QUERY PARAMETERS --------------------

/**
 * Query parameters for anomalies list
 */
export interface AnomaliesQueryParams {
  /** Start time range (ISO 8601) */
  start_date?: string
  /** End time range (ISO 8601) */
  end_date?: string
  /** Filter by severity level */
  severity?: AnomalySeverity
  /** Page number (min: 1) */
  page?: number
  /** Items per page (1-100) */
  per_page?: number
}

// -------------------- WEBSOCKET MESSAGES --------------------

/**
 * WebSocket event: new sensor reading
 */
export interface WSNewSensorReading {
  type: 'sensor_reading'
  data: SensorReading
}

/**
 * WebSocket event: new anomaly detected
 */
export interface WSNewAnomaly {
  type: 'anomaly_detected'
  data: MLPredictionResponse
}

/**
 * WebSocket event: system status update
 */
export interface WSSystemStatusUpdate {
  type: 'system_status_update'
  data: SystemStatus
}

/**
 * All possible WebSocket message types
 */
export type WSMessage =
  | WSNewSensorReading
  | WSNewAnomaly
  | WSSystemStatusUpdate

// -------------------- UTILITY TYPES --------------------

/**
 * API Request options
 */
export interface ApiRequestOptions {
  headers?: Record<string, string>
  params?: Record<string, any>
  signal?: AbortSignal
}

/**
 * Async state for pending/loading operations
 */
export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: ErrorResponse | null
}

// -------------------- HELPER FUNCTIONS --------------------

/**
 * Check if response is error
 */
export function isErrorResponse(response: any): response is ErrorResponse {
  return response && typeof response === 'object' && 'error' in response
}

/**
 * Extract error message from various error types
 */
export function getErrorMessage(error: ErrorResponse | Error | unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  if (isErrorResponse(error)) {
    return error.error.message
  }
  return 'Unknown error occurred'
}

/**
 * Format anomaly score for UI display
 */
export function formatAnomalyScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

/**
 * Get Tailwind severity color classes
 */
export function getSeverityColor(severity: AnomalySeverity): string {
  const colors: Record<AnomalySeverity, string> = {
    [AnomalySeverity.Normal]: 'text-green-600 bg-green-50',
    [AnomalySeverity.Warning]: 'text-yellow-600 bg-yellow-50',
    [AnomalySeverity.Critical]: 'text-red-600 bg-red-50'
  }
  return colors[severity] || 'text-gray-600 bg-gray-50'
}

/**
 * Get component status color classes
 */
export function getComponentStatusColor(status: ComponentStatus): string {
  const colors: Record<ComponentStatus, string> = {
    [ComponentStatus.Normal]: 'text-green-600 bg-green-50',
    [ComponentStatus.Degraded]: 'text-yellow-600 bg-yellow-50',
    [ComponentStatus.Failed]: 'text-red-600 bg-red-50'
  }
  return colors[status] || 'text-gray-600 bg-gray-50'
}

/**
 * Get severity icon name
 */
export function getSeverityIcon(severity: AnomalySeverity): string {
  const icons: Record<AnomalySeverity, string> = {
    [AnomalySeverity.Normal]: 'heroicons:check-circle',
    [AnomalySeverity.Warning]: 'heroicons:exclamation-triangle',
    [AnomalySeverity.Critical]: 'heroicons:x-circle'
  }
  return icons[severity] || 'heroicons:information-circle'
}

// -------------------- TYPE GUARDS --------------------

/**
 * Type guard: Validate sensor reading
 */
export function isValidSensorReading(data: any): data is SensorReading {
  return (
    data &&
    typeof data === 'object' &&
    typeof data.sensor_id === 'string' &&
    typeof data.timestamp === 'string' &&
    typeof data.value === 'number' &&
    Object.values(SensorUnit).includes(data.unit)
  )
}

/**
 * Type guard: Validate ML prediction
 */
export function isValidMLPrediction(data: any): data is MLPredictionResponse {
  return (
    data &&
    typeof data === 'object' &&
    typeof data.prediction_id === 'string' &&
    typeof data.system_id === 'string' &&
    typeof data.anomaly_score === 'number' &&
    Object.values(AnomalySeverity).includes(data.severity)
  )
}

/**
 * Type guard: Validate WebSocket message
 */
export function isValidWSMessage(data: any): data is WSMessage {
  return (
    data &&
    typeof data === 'object' &&
    typeof data.type === 'string' &&
    ['sensor_reading', 'anomaly_detected', 'system_status_update'].includes(data.type)
  )
}
