/**
 * Diagnosis API Types
 * Full TypeScript definitions for diagnosis flow with RAG integration
 * @version 1.0.0
 */

// ==================== CORE TYPES ====================

export type DiagnosisStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type Severity = 'normal' | 'warning' | 'critical'
export type RecommendationPriority = 'high' | 'medium' | 'low'
export type EquipmentType = 'pump' | 'valve' | 'cylinder' | 'motor' | 'accumulator'

// ==================== DIAGNOSIS RESULT ====================

/**
 * Complete diagnosis result with GNN predictions and RAG interpretation
 */
export interface DiagnosisResultResponse {
  session_id: string
  status: DiagnosisStatus
  created_at: string // ISO 8601
  completed_at: string | null
  
  /** GNN Model Output */
  gnn_result: GNNResult
  
  /** RAG Interpretation (null if not ready or failed) */
  rag_interpretation: RAGInterpretation | null
  
  /** Equipment Context */
  equipment_context: EquipmentContext
  
  /** Telemetry data snapshot (optional) */
  telemetry_snapshot?: TelemetrySnapshot
}

// ==================== GNN RESULT ====================

export interface GNNResult {
  predictions: MLPrediction[]
  anomaly_score: number // 0.0 - 1.0
  overall_severity: Severity
  model_version: string
  inference_time_ms: number
  confidence: number // 0.0 - 1.0
}

export interface MLPrediction {
  component: string
  fault_type: string | null
  probability: number // 0.0 - 1.0
  severity: Severity
  anomaly_indicators: AnomalyIndicators
}

export interface AnomalyIndicators {
  pressure: number | null
  temperature: number | null
  vibration: number | null
  flow_rate: number | null
}

// ==================== RAG INTERPRETATION ====================

export interface RAGInterpretation {
  /** Brief summary (max 250 chars) */
  summary: string
  
  /** Structured step-by-step reasoning */
  reasoning: ReasoningStep[]
  
  /** Prioritized recommendations */
  recommendations: Recommendation[]
  
  severity: Severity
  confidence: number // 0.0 - 1.0
  
  /** Prognosis (can be null if model uncertain) */
  prognosis: string | null
  
  /** Technical metadata */
  metadata: RAGMetadata
}

export interface ReasoningStep {
  step: number
  title: string
  description: string
  evidence: string[]
  conclusion: string
}

export interface Recommendation {
  priority: RecommendationPriority
  action: string
  rationale: string
  estimated_time: string | null
  requires_shutdown: boolean
  parts_needed: string[] | null
}

export interface RAGMetadata {
  model_version: string
  processing_time_ms: number
  tokens_used: number
  temperature: number
  rag_sources: number
}

// ==================== EQUIPMENT CONTEXT ====================

export interface EquipmentContext {
  id: string
  name: string
  type: EquipmentType
  operating_hours: number
  last_maintenance: string | null // ISO 8601
  maintenance_interval: number // hours
  location: string | null
}

// ==================== TELEMETRY ====================

export interface TelemetrySnapshot {
  pressure: number[]
  temperature: number[]
  flow_rate: number[]
  timestamps: string[] // ISO 8601
}

// ==================== WEBSOCKET MESSAGES ====================

export type WSMessage = 
  | ProgressUpdate
  | StageUpdate
  | ResultUpdate
  | ErrorUpdate

export interface ProgressUpdate {
  type: 'progress'
  stage: 'gnn_inference' | 'rag_generation' | 'finalizing'
  progress: number // 0-100
  message: string
  timestamp: string
}

export interface StageUpdate {
  type: 'stage_complete'
  stage: 'gnn_inference' | 'rag_generation'
  duration_ms: number
  timestamp: string
  partial_result?: {
    gnn_result?: GNNResult
    rag_interpretation?: RAGInterpretation
  }
}

export interface ResultUpdate {
  type: 'result_complete'
  result: DiagnosisResultResponse
  timestamp: string
}

export interface ErrorUpdate {
  type: 'error'
  stage: string
  error_code: string
  message: string
  details?: Record<string, any>
  timestamp: string
}

// ==================== ERROR HANDLING ====================

export type ErrorCode = 
  | 'GNN_INFERENCE_FAILED'
  | 'RAG_GENERATION_TIMEOUT'
  | 'INVALID_SESSION'
  | 'EQUIPMENT_NOT_FOUND'

export interface ErrorResponse {
  error_code: ErrorCode
  message: string
  details?: {
    stage: string
    original_error: string
    retry_possible: boolean
  }
  timestamp: string
}

// ==================== HELPER FUNCTIONS ====================

/**
 * Get severity color class (Tailwind)
 */
export function getSeverityColor(severity: Severity): string {
  const colors: Record<Severity, string> = {
    normal: 'text-green-600 bg-green-50 border-green-200',
    warning: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    critical: 'text-red-600 bg-red-50 border-red-200'
  }
  return colors[severity]
}

/**
 * Get severity icon name
 */
export function getSeverityIcon(severity: Severity): string {
  const icons: Record<Severity, string> = {
    normal: 'lucide:check-circle',
    warning: 'lucide:alert-triangle',
    critical: 'lucide:alert-octagon'
  }
  return icons[severity]
}

/**
 * Get priority color
 */
export function getPriorityColor(priority: RecommendationPriority): string {
  const colors: Record<RecommendationPriority, string> = {
    high: 'text-red-600 bg-red-50 border-red-200',
    medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    low: 'text-blue-600 bg-blue-50 border-blue-200'
  }
  return colors[priority]
}

/**
 * Format confidence as percentage
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}

/**
 * Format anomaly score as percentage
 */
export function formatAnomalyScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

/**
 * Format duration (ms to human-readable)
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
}

// ==================== TYPE GUARDS ====================

/**
 * Check if diagnosis has RAG interpretation
 */
export function hasRAGInterpretation(
  result: DiagnosisResultResponse
): result is DiagnosisResultResponse & { rag_interpretation: RAGInterpretation } {
  return result.rag_interpretation !== null
}

/**
 * Check if WebSocket message is error
 */
export function isErrorUpdate(message: WSMessage): message is ErrorUpdate {
  return message.type === 'error'
}

/**
 * Check if WebSocket message is progress
 */
export function isProgressUpdate(message: WSMessage): message is ProgressUpdate {
  return message.type === 'progress'
}

/**
 * Check if WebSocket message is stage complete
 */
export function isStageUpdate(message: WSMessage): message is StageUpdate {
  return message.type === 'stage_complete'
}

/**
 * Check if WebSocket message is final result
 */
export function isResultUpdate(message: WSMessage): message is ResultUpdate {
  return message.type === 'result_complete'
}
