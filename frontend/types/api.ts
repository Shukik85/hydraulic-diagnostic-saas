/**
 * TypeScript Types для Hydraulic Diagnostic Platform API
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
 * Единицы измерения датчиков
 */
export enum SensorUnit {
  Bar = 'bar',
  Celsius = 'celsius',
  RPM = 'rpm',
  LPM = 'lpm'
}

/**
 * Предпочтения ML моделей
 */
export enum ModelPreference {
  CatBoost = 'catboost',
  XGBoost = 'xgboost',
  RandomForest = 'random_forest',
  Adaptive = 'adaptive',
  Ensemble = 'ensemble'
}

/**
 * Уровень критичности аномалии
 */
export enum AnomalySeverity {
  Normal = 'normal',
  Warning = 'warning',
  Critical = 'critical'
}

/**
 * Типы компонентов системы
 */
export enum ComponentType {
  Pump = 'pump',
  Valve = 'valve',
  Accumulator = 'accumulator',
  Cooler = 'cooler'
}

/**
 * Статус компонента
 */
export enum ComponentStatus {
  Normal = 'normal',
  Degraded = 'degraded',
  Failed = 'failed'
}

/**
 * Статус обработки задания
 */
export enum JobStatus {
  Queued = 'queued',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed'
}

/**
 * Коды ошибок API
 */
export enum ErrorCode {
  ValidationError = 'VALIDATION_ERROR',
  AuthenticationError = 'AUTHENTICATION_ERROR',
  RateLimitExceeded = 'RATE_LIMIT_EXCEEDED',
  InternalError = 'INTERNAL_ERROR',
  MLServiceUnavailable = 'ML_SERVICE_UNAVAILABLE',
  NotFound = 'NOT_FOUND',
  Forbidden = 'FORBIDDEN'
}

// -------------------- REQUEST SCHEMAS --------------------

/**
 * Показания одного датчика
 */
export interface SensorReading {
  /** UUID датчика */
  sensor_id: string
  /** ISO 8601 timestamp */
  timestamp: string
  /** Численное значение показания */
  value: number
  /** Единица измерения */
  unit: SensorUnit
  /** Качество данных (0-100%) */
  quality?: number
}

/**
 * Массовая загрузка данных датчиков
 */
export interface SensorBulkIngest {
  /** UUID системы */
  system_id: string
  /** Массив показаний (1-10000) */
  readings: SensorReading[]
}

/**
 * Запрос ML предсказания
 */
export interface MLPredictionRequest {
  /** UUID системы */
  system_id: string
  /** Данные датчиков для анализа */
  sensor_data: SensorReading[]
  /** Предпочитаемая модель (default: ensemble) */
  model_preference?: ModelPreference
}

// -------------------- RESPONSE SCHEMAS --------------------

/**
 * Ответ после массовой загрузки
 */
export interface BulkIngestResponse {
  /** UUID задания обработки */
  job_id: string
  /** Статус обработки */
  status: JobStatus
}

/**
 * Детали предсказания одной модели
 */
export interface ModelPrediction {
  /** Название модели */
  model: ModelPreference
  /** Скор аномальности (0-1) */
  score: number
  /** Уверенность модели (0-1) */
  confidence: number
}

/**
 * Метаданные ML inference
 */
export interface PredictionMetadata {
  /** Время инференса в миллисекундах */
  inference_time_ms: number
  /** Версия модели */
  model_version: string
  /** Использован ли кэш */
  cache_hit: boolean
}

/**
 * Ответ ML предсказания
 */
export interface MLPredictionResponse {
  /** UUID предсказания */
  prediction_id: string
  /** UUID системы */
  system_id: string
  /** ISO 8601 timestamp */
  timestamp: string
  /** Общий скор аномальности (0-1) */
  anomaly_score: number
  /** Уровень критичности */
  severity: AnomalySeverity
  /** Предсказания от разных моделей */
  predictions: ModelPrediction[]
  /** Метаданные инференса */
  metadata: PredictionMetadata
}

/**
 * Статус компонента системы
 */
export interface ComponentStatusDetail {
  /** ID компонента */
  component_id: string
  /** Тип компонента */
  type: ComponentType
  /** Текущий статус */
  status: ComponentStatus
  /** Время последней аномалии */
  last_anomaly: string | null
}

/**
 * Общий статус системы
 */
export interface SystemStatus {
  /** UUID системы */
  system_id: string
  /** Общий health score (0-100) */
  health_score: number
  /** Статусы компонентов */
  component_statuses: ComponentStatusDetail[]
  /** Время последнего обновления */
  last_updated: string
}

/**
 * Пагинация
 */
export interface Pagination {
  /** Текущая страница */
  page: number
  /** Элементов на странице */
  per_page: number
  /** Всего элементов */
  total: number
  /** Всего страниц */
  pages: number
}

/**
 * Список аномалий с пагинацией
 */
export interface AnomaliesListResponse {
  /** Массив аномалий */
  items: MLPredictionResponse[]
  /** Информация о пагинации */
  pagination: Pagination
}

// -------------------- ERROR HANDLING --------------------

/**
 * Детали ошибки API
 */
export interface APIErrorDetail {
  /** Код ошибки */
  code: ErrorCode
  /** Человекочитаемое сообщение */
  message: string
  /** Дополнительные детали */
  details?: Record<string, any>
  /** ISO 8601 timestamp */
  timestamp: string
  /** UUID запроса для трейсинга */
  request_id: string
}

/**
 * Стандартный формат ответа с ошибкой
 */
export interface ErrorResponse {
  error: APIErrorDetail
}

// -------------------- QUERY PARAMETERS --------------------

/**
 * Параметры запроса списка аномалий
 */
export interface AnomaliesQueryParams {
  /** Начало диапазона времени (ISO 8601) */
  start_date?: string
  /** Конец диапазона времени (ISO 8601) */
  end_date?: string
  /** Фильтр по уровню критичности */
  severity?: AnomalySeverity
  /** Номер страницы (min: 1) */
  page?: number
  /** Элементов на странице (1-100) */
  per_page?: number
}

// -------------------- WEBSOCKET MESSAGES --------------------

/**
 * WebSocket событие: новое показание датчика
 */
export interface WSNewSensorReading {
  type: 'sensor_reading'
  data: SensorReading
}

/**
 * WebSocket событие: новая аномалия
 */
export interface WSNewAnomaly {
  type: 'anomaly_detected'
  data: MLPredictionResponse
}

/**
 * WebSocket событие: обновление статуса системы
 */
export interface WSSystemStatusUpdate {
  type: 'system_status_update'
  data: SystemStatus
}

/**
 * Все возможные типы WebSocket сообщений
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
 * Тип для pending/loading состояний
 */
export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: ErrorResponse | null
}

// -------------------- HELPER FUNCTIONS --------------------

/**
 * Helper: Check if response is error
 */
export function isErrorResponse(response: any): response is ErrorResponse {
  return response && typeof response === 'object' && 'error' in response
}

/**
 * Helper: Extract error message
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
 * Helper: Format anomaly score для UI
 */
export function formatAnomalyScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

/**
 * Helper: Get severity color (Tailwind classes)
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
 * Helper: Get component status color
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
 * Helper: Get severity badge icon
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
 * Type guard: Check if SensorReading is valid
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
 * Type guard: Check if MLPredictionResponse is valid
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
 * Type guard: Check if WebSocket message is valid
 */
export function isValidWSMessage(data: any): data is WSMessage {
  return (
    data &&
    typeof data === 'object' &&
    typeof data.type === 'string' &&
    ['sensor_reading', 'anomaly_detected', 'system_status_update'].includes(data.type)
  )
}
