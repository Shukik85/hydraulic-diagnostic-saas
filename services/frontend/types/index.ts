/**
 * Central Types Index
 * Re-exports all type definitions for easy import
 * @version 1.0.0
 */

// ==================== API TYPES ====================
export type {
  User,
  LoginCredentials,
  RegisterData,
  AuthTokens,
  ApiResponse,
  HydraulicSystem,
  DiagnosticSession,
  ApiError,
  TableColumn,
  UiPasswordStrength,
  PasswordStrength
} from './api'

// ==================== CHAT TYPES ====================
export type {
  ChatRole,
  ChatMessage,
  ChatSession
} from './chat'

// ==================== DIAGNOSIS TYPES ====================
export type {
  DiagnosisStatus,
  Severity,
  RecommendationPriority,
  EquipmentType,
  DiagnosisResultResponse,
  GNNResult,
  MLPrediction,
  AnomalyIndicators,
  RAGInterpretation,
  ReasoningStep,
  Recommendation,
  RAGMetadata,
  EquipmentContext,
  TelemetrySnapshot,
  WSMessage,
  ProgressUpdate,
  StageUpdate,
  ResultUpdate,
  ErrorUpdate,
  ErrorCode,
  ErrorResponse
} from './diagnosis'

// Export diagnosis utility functions
export {
  getSeverityColor,
  getSeverityIcon,
  getPriorityColor,
  formatConfidence,
  formatAnomalyScore,
  formatDuration,
  hasRAGInterpretation,
  isErrorUpdate,
  isProgressUpdate,
  isStageUpdate,
  isResultUpdate
} from './diagnosis'

// ==================== METADATA TYPES ====================
export type {
  ComponentType,
  EquipmentType as MetadataEquipmentType,
  NormalRange,
  ComponentMetadata,
  PumpSpecific,
  MotorSpecific,
  CylinderSpecific,
  ValveSpecific,
  FilterSpecific,
  AccumulatorSpecific,
  SystemMetadata,
  DutyCycle,
  SensorConfig,
  SensorDetails,
  IncompleteDataReport,
  WizardState,
  ValidationResult,
  SmartDefault
} from './metadata'

// ==================== RAG TYPES ====================
export type {
  RAGInterpretationRequest,
  RAGInterpretationResponse,
  KnowledgeDocument,
  KnowledgeBaseSearchRequest,
  KnowledgeBaseSearchResponse,
  ConfidenceLevel,
  RAGStatus
} from './rag'

// ==================== TYPE UTILITIES ====================

/**
 * Generic async state type
 */
export type AsyncState<T> = {
  data: T | null
  loading: boolean
  error: string | null
}

/**
 * System status type
 */
export type SystemStatus = 'online' | 'offline' | 'maintenance' | 'error'

/**
 * Pagination params
 */
export interface PaginationParams {
  page?: number
  limit?: number
  offset?: number
}

/**
 * Sort params
 */
export interface SortParams {
  field: string
  direction: 'asc' | 'desc'
}

/**
 * Filter params
 */
export interface FilterParams {
  [key: string]: string | number | boolean | null | undefined
}
