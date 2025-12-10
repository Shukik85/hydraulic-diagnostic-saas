/**
 * TypeScript Types для Hydraulic Diagnostic Platform API
 *
 * ✅ УПдЕЙТЭД:
 * - Новые типы для paginated responses
 * - Новые типы для error handling
 * - Энумы более строгие
 * - Удалены ChatMessage, ChatSession (теперь в rag.ts)
 *
 * @see https://github.com/Shukik85/hydraulic-diagnostic-saas
 * @version 2.0.0
 */

// ==================== USER & AUTH ====================

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

export interface AuthResponse {
  user: User
  access: string
  refresh?: string
}

// ==================== HYDRAULIC SYSTEMS ====================

export type SystemType = 'industrial' | 'mobile' | 'marine' | 'construction' | 'mining' | 'agricultural'

export type SystemStatus = 'active' | 'maintenance' | 'inactive'

export interface HydraulicSystem {
  id: number
  name: string
  type: SystemType
  status: SystemStatus
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

export interface SystemSummary {
  id: number
  name: string
  type: SystemType
  status: SystemStatus
  health_score: number
}

export interface CreateSystemRequest {
  name: string
  type: SystemType
  description?: string
  location?: string
}

export interface UpdateSystemRequest extends Partial<CreateSystemRequest> {}

// ==================== DIAGNOSTICS ====================

export type DiagnosticType = 'full' | 'pressure' | 'temperature' | 'vibration' | 'flow'

export type DiagnosticStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface DiagnosticSession {
  id: number
  system_id: number
  type: DiagnosticType
  status: DiagnosticStatus
  progress: number
  started_at: string
  completed_at?: string
  results?: DiagnosticResult[]
  created_by: number
}

export interface DiagnosticResult {
  id: number
  session_id: number
  metric: string
  value: number
  unit: string
  status: 'normal' | 'warning' | 'critical'
  timestamp: string
}

export interface StartDiagnosticRequest {
  system_id: number
  type: DiagnosticType
}

// ==================== PAGINATION & LIST RESPONSES ====================

/**
 * Генеричная структура для paginated responses
 * Используется чтобы стандартизировать работу с list эндпоинтами
 */
export interface PaginatedResponse<T> {
  results: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface ListRequest {
  page?: number
  page_size?: number
  search?: string
  order_by?: string
  filters?: Record<string, any>
}

// ==================== ERROR HANDLING ====================

export interface ApiError {
  code?: string
  message: string
  status: number
  details?: Record<string, any>
  data?: any
}

export interface ValidationError {
  field: string
  code: string
  message: string
}

export interface ValidationErrorResponse {
  errors: ValidationError[]
  message?: string
}

export interface ApiErrorResponse {
  error: string
  message?: string
  details?: Record<string, any>
}

// ==================== GENERIC RESPONSES ====================

export interface ApiResponse<T = any> {
  data: T
  message?: string
  error?: string
}

export interface SuccessResponse<T = any> {
  status: 'success'
  data: T
  message?: string
}

export interface ErrorResponse {
  status: 'error'
  error: string
  message?: string
  details?: Record<string, any>
}

// ==================== UI & HELPERS ====================

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

// ==================== CONSTANTS & TYPE GUARDS ====================

export const SYSTEM_TYPE_LABELS: Record<SystemType, string> = {
  industrial: 'Industrial',
  mobile: 'Mobile',
  marine: 'Marine',
  construction: 'Construction',
  mining: 'Mining',
  agricultural: 'Agricultural',
}

export const SYSTEM_STATUS_LABELS: Record<SystemStatus, string> = {
  active: 'Активна',
  maintenance: 'Maintenance',
  inactive: 'Неактивна',
}

export const DIAGNOSTIC_STATUS_LABELS: Record<DiagnosticStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
}

// ==================== TYPE GUARDS ====================

export function isHydraulicSystem(obj: any): obj is HydraulicSystem {
  return obj && typeof obj === 'object' && 'id' in obj && 'name' in obj && 'pressure' in obj
}

export function isUser(obj: any): obj is User {
  return obj && typeof obj === 'object' && 'email' in obj && 'id' in obj
}

export function isApiError(obj: any): obj is ApiError {
  return obj && typeof obj === 'object' && 'message' in obj && 'status' in obj
}

export function isPaginatedResponse<T>(obj: any): obj is PaginatedResponse<T> {
  return (
    obj &&
    typeof obj === 'object' &&
    Array.isArray(obj.results) &&
    typeof obj.total === 'number'
  )
}
