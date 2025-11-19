/**
 * Типы для системы диагностики гидравлических систем
 */

export type DiagnosticStatus = 'normal' | 'warning' | 'critical' | 'unknown'

export interface ModelPrediction {
  model: string
  prediction: string
  confidence: number
}

export interface Anomaly {
  parameter: string
  description: string
  severity: string
  value: number | string
  expectedRange: string
  deviation: number
}

export interface FeatureImportance {
  name: string
  importance: number
}

export interface DiagnosticResult {
  id?: number | string
  timestamp: string
  status: DiagnosticStatus
  confidence: number
  predictions: ModelPrediction[]
  anomalies: Anomaly[]
  featureImportance: FeatureImportance[]
  equipmentId?: string
  equipmentName?: string
}

export interface DiagnosticHistoryItem {
  id: number | string
  timestamp: string
  status: DiagnosticStatus
  confidence: number
  anomalyCount: number
  modelCount: number
  equipmentName?: string
  tags?: string[]
}
