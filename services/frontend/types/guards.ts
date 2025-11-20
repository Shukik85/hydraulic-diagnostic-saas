/**
 * guards.ts - Type Guards для runtime безопасности
 * Enterprise стандарт: используйте type guards вместо type assertions
 */

import type {
  SystemStatus,
  ErrorResponse,
  AnomaliesListResponse,
  ComponentStatus
} from './api'

import type {
  RAGInterpretationResponse,
  KnowledgeBaseSearchResponse
} from './rag'

/**
 * Проверяет является ли объект ErrorResponse
 */
export function isErrorResponse(obj: unknown): obj is ErrorResponse {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'error' in obj &&
    typeof (obj as any).error === 'object' &&
    'message' in (obj as any).error
  )
}

/**
 * Проверяет является ли объект SystemStatus
 */
export function isSystemStatus(obj: unknown): obj is SystemStatus {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'health_score' in obj &&
    'component_statuses' in obj &&
    typeof (obj as any).health_score === 'number' &&
    Array.isArray((obj as any).component_statuses)
  )
}

/**
 * Проверяет является ли объект AnomaliesListResponse
 */
export function isAnomaliesListResponse(obj: unknown): obj is AnomaliesListResponse {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'items' in obj &&
    Array.isArray((obj as any).items) &&
    'pagination' in obj
  )
}

/**
 * Проверяет является ли объект RAGInterpretationResponse
 */
export function isRAGInterpretationResponse(obj: unknown): obj is RAGInterpretationResponse {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'reasoning' in obj &&
    'summary' in obj &&
    'analysis' in obj &&
    'recommendations' in obj &&
    'confidence' in obj &&
    'knowledgeUsed' in obj &&
    Array.isArray((obj as any).recommendations) &&
    Array.isArray((obj as any).knowledgeUsed)
  )
}

/**
 * Проверяет является ли объект KnowledgeBaseSearchResponse
 */
export function isKnowledgeBaseSearchResponse(obj: unknown): obj is KnowledgeBaseSearchResponse {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'documents' in obj &&
    'totalResults' in obj &&
    Array.isArray((obj as any).documents) &&
    typeof (obj as any).totalResults === 'number'
  )
}

/**
 * Проверяет является ли объект ComponentStatus
 */
export function isComponentStatus(obj: unknown): obj is ComponentStatus {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'component_id' in obj &&
    'name' in obj &&
    'status' in obj
  )
}
