/**
 * Extended API Types
 * 
 * Additional types for working with generated API client.
 * These complement the auto-generated types from OpenAPI.
 */

import type { System, Component, Sensor } from '~/generated/api'

// ==================== API RESPONSE TYPES ====================

/**
 * Generic API response wrapper
 */
export interface ApiResponse<T> {
  data: T
  status: number
  statusText: string
  headers: Record<string, string>
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

/**
 * API error response
 */
export interface ApiError {
  detail: string
  status: number
  type?: string
  errors?: Record<string, string[]>
}

// ==================== SENSOR DATA ====================

/**
 * Sensor reading for CSV upload
 */
export interface SensorReading {
  system_id: string
  sensor_id: string
  timestamp: string
  value: number
  unit: string
  quality?: number
  sensor_type?: string
  component_id?: string
}

/**
 * Batch sensor readings
 */
export interface BatchSensorReadings {
  system_id: string
  readings: SensorReading[]
  source: 'csv' | 'api' | 'manual'
  uploaded_by?: string
}

// ==================== DIAGNOSIS ====================

/**
 * Diagnosis status
 */
export type DiagnosisStatus = 
  | 'pending'
  | 'running'
  | 'gnn_complete'
  | 'rag_processing'
  | 'complete'
  | 'failed'

/**
 * Extended diagnosis with RAG
 */
export interface DiagnosisWithRAG {
  id: string
  system_id: string
  status: DiagnosisStatus
  created_at: string
  completed_at?: string
  gnn_result?: any
  rag_interpretation?: any
  error?: string
}

// ==================== REAL-TIME UPDATES ====================

/**
 * WebSocket message types
 */
export type WebSocketMessageType =
  | 'sensor_reading'
  | 'system_status_update'
  | 'diagnosis_progress'
  | 'diagnosis_complete'
  | 'alert'

/**
 * WebSocket message
 */
export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType
  data: T
  timestamp: string
}

/**
 * System status update
 */
export interface SystemStatusUpdate {
  system_id: string
  status: 'online' | 'offline' | 'maintenance' | 'error'
  message?: string
  updated_at: string
}

/**
 * Diagnosis progress update
 */
export interface DiagnosisProgressUpdate {
  diagnosis_id: string
  stage: 'gnn' | 'rag'
  progress: number  // 0-100
  message: string
}

// ==================== UI STATE ====================

/**
 * Loading state
 */
export interface LoadingState {
  loading: boolean
  error: string | null
  retryCount: number
}

/**
 * Form state
 */
export interface FormState<T> {
  data: T
  errors: Record<keyof T, string>
  dirty: boolean
  submitting: boolean
}

// ==================== FILTERS ====================

/**
 * System filters
 */
export interface SystemFilters {
  search?: string
  status?: 'online' | 'offline' | 'maintenance' | 'error'
  equipment_type?: string
  manufacturer?: string
  location?: string
}

/**
 * Diagnosis filters
 */
export interface DiagnosisFilters {
  system_id?: string
  status?: DiagnosisStatus
  date_from?: string
  date_to?: string
  anomaly_threshold?: number
}

// ==================== TYPE GUARDS ====================

/**
 * Check if response is error
 */
export function isApiError(response: any): response is ApiError {
  return response && typeof response.detail === 'string'
}

/**
 * Check if response is paginated
 */
export function isPaginatedResponse<T>(response: any): response is PaginatedResponse<T> {
  return response && Array.isArray(response.items) && typeof response.total === 'number'
}

// ==================== HELPER TYPES ====================

/**
 * Extract array item type
 */
export type ArrayItem<T> = T extends (infer U)[] ? U : never

/**
 * Make all properties optional
 */
export type PartialDeep<T> = {
  [P in keyof T]?: T[P] extends object ? PartialDeep<T[P]> : T[P]
}

/**
 * Make specific properties required
 */
export type RequireKeys<T, K extends keyof T> = T & Required<Pick<T, K>>
