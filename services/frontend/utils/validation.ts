// utils/validation.ts
import type { ReasoningStep, Recommendation, Severity } from '../types/diagnosis'

export function validateConfidence(confidence: number): number {
  if (typeof confidence !== 'number' || isNaN(confidence)) return 0
  return Math.max(0, Math.min(1, confidence))
}

export function validateSeverity(severity: unknown): Severity {
  const valid: Severity[] = ['normal','warning','critical']
  if (valid.includes(severity as Severity)) return severity as Severity
  return 'normal'
}

export function validateRecommendations(recs: unknown): Recommendation[] {
  if (!Array.isArray(recs)) return []
  // Проверить наличие необходимых полей и типов
  return recs.filter(r=>r && typeof r.priority==='string' && typeof r.action==='string') as Recommendation[]
}

export function validateReasoning(reasoning: unknown): ReasoningStep[]|string {
  if (Array.isArray(reasoning) && reasoning.every(step => typeof step.step === 'number' && typeof step.title==='string')) {
    return reasoning as ReasoningStep[]
  }
  if (typeof reasoning === 'string') return reasoning
  return ''
}
