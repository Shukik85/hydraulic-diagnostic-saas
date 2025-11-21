// types/rag.ts
/**
 * TypeScript types для RAG (Retrieval-Augmented Generation) Service
 */

/**
 * Запрос на интерпретацию GNN результатов.
 */
export interface RAGInterpretationRequest {
  /**
   * Результаты GNN диагностики
   */
  gnnResults: any  // TODO: добавить proper GNN types
  
  /**
   * ID оборудования
   */
  equipmentId: string
  
  /**
   * Контекст оборудования (опционально)
   */
  equipmentContext?: {
    name?: string
    type?: string
    manufacturer?: string
    model?: string
    installDate?: string
    operatingHours?: number
    maintenanceHistory?: string[]
  }
  
  /**
   * Использовать ли Knowledge Base
   */
  useKnowledgeBase?: boolean
  
  /**
   * Максимальное количество токенов для генерации
   */
  maxTokens?: number
}

/**
 * Шаг рассуждения (Reasoning Step)
 */
export interface ReasoningStep {
  /**
   * Заголовок шага
   */
  title: string
  
  /**
   * Описание шага
   */
  description: string
  
  /**
   * Доказательства/ключевые точки
   */
  evidence: string[]
}

/**
 * Ответ от RAG Service с интерпретацией.
 */
export interface RAGInterpretationResponse {
  /**
   * Reasoning process - пошаговое мышление модели
   * (внутренний монолог DeepSeek-R1)
   */
  reasoning: string
  
  /**
   * Executive summary - краткая сводка
   */
  summary: string
  
  /**
   * Detailed analysis - детальный анализ
   */
  analysis: string
  
  /**
   * Рекомендации по действиям
   */
  recommendations: string[]
  
  /**
   * Confidence score (0-1)
   */
  confidence: number
  
  /**
   * Использованные документы из Knowledge Base
   */
  knowledgeUsed: KnowledgeDocument[]
  
  /**
   * Метаданные генерации
   */
  metadata: {
    model: string
    processingTime: number
    tokensUsed: number
  }
}

/**
 * Документ из Knowledge Base.
 */
export interface KnowledgeDocument {
  /**
   * Unique document ID
   */
  id: string
  
  /**
   * Титул документа
   */
  title: string
  
  /**
   * Содержимое (часть или chunk)
   */
  content: string
  
  /**
   * Relevance score (0-1)
   */
  score: number
  
  /**
   * Метаданные документа
   */
  metadata?: {
    source?: string
    author?: string
    category?: string
    tags?: string[]
    createdAt?: string
    updatedAt?: string
  }
}

/**
 * Запрос на поиск в Knowledge Base.
 */
export interface KnowledgeBaseSearchRequest {
  /**
   * Поисковый запрос
   */
  query: string
  
  /**
   * Количество результатов
   */
  topK?: number
  
  /**
   * Фильтры
   */
  filters?: {
    category?: string
    tags?: string[]
    dateFrom?: string
    dateTo?: string
  }
  
  /**
   * Минимальный score для возврата
   */
  minScore?: number
}

/**
 * Ответ от Knowledge Base search.
 */
export interface KnowledgeBaseSearchResponse {
  /**
   * Найденные документы
   */
  documents: KnowledgeDocument[]
  
  /**
   * Общее количество результатов
   */
  totalResults: number
  
  /**
   * Время поиска (мс)
   */
  searchTime: number
}

/**
 * Confidence level интерпретации.
 */
export type ConfidenceLevel = 'high' | 'medium' | 'low'

/**
 * Статус RAG генерации.
 */
export type RAGStatus = 'idle' | 'loading' | 'streaming' | 'completed' | 'error'
