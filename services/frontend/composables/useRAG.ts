/**
 * useRAG.ts — Composable для RAG (Retrieval-Augmented Generation)
 * DeepSeek-R1 интеграция с Knowledge Base
 */
import { ref } from 'vue'
import type {
  RAGInterpretationRequest,
  RAGInterpretationResponse,
  KnowledgeBaseSearchRequest,
  KnowledgeBaseSearchResponse,
  RAGStatus
} from '~/types/rag'

/**
 * Confidence Level объект
 */
export interface ConfidenceLevel {
  level: 'high' | 'medium' | 'low'
  label: string
  color: string
}

/**
 * Utility: Определить уровень уверенности
 * @param confidence - Значение confidence (0-1)
 * @returns Объект с level, label, color
 */
export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.8) {
    return {
      level: 'high',
      label: 'Высокая',
      color: 'green'
    }
  }
  
  if (confidence >= 0.5) {
    return {
      level: 'medium',
      label: 'Средняя',
      color: 'yellow'
    }
  }
  
  return {
    level: 'low',
    label: 'Низкая',
    color: 'red'
  }
}

export function useRAG() {
  const status = ref<RAGStatus>('idle')
  const error = ref<string | null>(null)

  /**
   * Проверка доступности RAG feature
   */
  function isRAGEnabled(): boolean {
    // TODO: Добавить проверку config когда будет реализовано на backend
    return true
  }

  /**
   * Получить интерпретацию результатов диагностики от RAG
   */
  async function interpretDiagnosis(
    _request: RAGInterpretationRequest
  ): Promise<RAGInterpretationResponse | null> {
    status.value = 'loading'
    error.value = null

    try {
      // TODO: Реализовать когда backend API будет готов
      // Временный mock для typecheck
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const mockResponse: RAGInterpretationResponse = {
        reasoning: 'Mock reasoning process',
        summary: 'Mock summary of the diagnosis',
        analysis: 'Mock detailed analysis',
        recommendations: ['Mock recommendation 1', 'Mock recommendation 2'],
        confidence: 0.85,
        knowledgeUsed: [],
        metadata: {
          model: 'mock-model',
          processingTime: 100,
          tokensUsed: 50
        }
      }

      status.value = 'completed'
      return mockResponse
    } catch (err) {
      status.value = 'error'
      error.value = String(err)
      return null
    }
  }

  /**
   * Поиск в Knowledge Base
   */
  async function searchKnowledge(
    _request: KnowledgeBaseSearchRequest
  ): Promise<KnowledgeBaseSearchResponse | null> {
    status.value = 'loading'
    error.value = null

    try {
      // TODO: Реализовать когда backend API будет готов
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const mockResponse: KnowledgeBaseSearchResponse = {
        documents: [],
        totalResults: 0,
        searchTime: 100
      }

      status.value = 'completed'
      return mockResponse
    } catch (err) {
      status.value = 'error'
      error.value = String(err)
      return null
    }
  }

  /**
   * Объяснить конкретную аномалию
   */
  async function explainAnomaly(_anomaly: any): Promise<RAGInterpretationResponse | null> {
    status.value = 'loading'
    error.value = null

    try {
      // TODO: Реализовать когда backend API будет готов
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const mockResponse: RAGInterpretationResponse = {
        reasoning: 'Mock anomaly reasoning',
        summary: 'Mock anomaly summary',
        analysis: 'Mock anomaly analysis',
        recommendations: [],
        confidence: 0.8,
        knowledgeUsed: [],
        metadata: {
          model: 'mock-model',
          processingTime: 100,
          tokensUsed: 50
        }
      }

      status.value = 'completed'
      return mockResponse
    } catch (err) {
      status.value = 'error'
      error.value = String(err)
      return null
    }
  }

  return {
    status,
    error,
    isRAGEnabled,
    interpretDiagnosis,
    searchKnowledge,
    explainAnomaly,
    getConfidenceLevel  // ✅ Экспортируем utility
  }
}
