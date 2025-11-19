// composables/useRAG.ts
/**
 * RAG (Retrieval-Augmented Generation) Integration
 * Использует DeepSeek-R1 (70B) для AI-интерпретации GNN результатов
 */
import { useRuntimeConfig } from 'nuxt/app'
import { ref, computed, readonly } from 'vue'
import { useGeneratedApi } from './useGeneratedApi'
import type {
  RAGInterpretationRequest,
  RAGInterpretationResponse,
  KnowledgeBaseSearchRequest,
  KnowledgeBaseSearchResponse,
} from '~/types/rag'

export interface UseRAGOptions {
  useKnowledgeBase?: boolean
  timeout?: number
  maxTokens?: number
}

export function useRAG(options: UseRAGOptions = {}) {
  const config = useRuntimeConfig()
  const api = useGeneratedApi()
  const loading = ref(false)
  const error = ref<Error | null>(null)
  const lastInterpretation = ref<RAGInterpretationResponse | null>(null)
  const isRAGEnabled = computed(() => {
    return config.public.features?.ragInterpretation === true
  })

  const interpretDiagnosis = async (
    request: RAGInterpretationRequest,
  ): Promise<RAGInterpretationResponse | null> => {
    if (!isRAGEnabled.value) {
      console.warn('RAG feature is disabled. Enable with NUXT_PUBLIC_ENABLE_RAG=true')
      return null
    }
    loading.value = true
    error.value = null
    try {
      const response = await api.rag.interpretDiagnosis({
        gnnResults: request.gnnResults,
        equipmentId: request.equipmentId,
        equipmentContext: request.equipmentContext,
        useKnowledgeBase: options.useKnowledgeBase ?? true,
        maxTokens: options.maxTokens || 2048,
      })
      lastInterpretation.value = response as RAGInterpretationResponse
      return response as RAGInterpretationResponse
    }
    catch (err: any) {
      console.error('RAG interpretation error:', err)
      error.value = err
      return {
        reasoning: 'Анализ недоступен. Проверьте подключение к RAG Service.',
        summary: 'Ошибка при генерации интерпретации',
        analysis: err.message || 'Неизвестная ошибка',
        recommendations: ['Проверьте доступность RAG Service'],
        confidence: 0,
        knowledgeUsed: [],
        metadata: {
          model: 'fallback',
          processingTime: 0,
          tokensUsed: 0,
        },
      }
    }
    finally {
      loading.value = false
    }
  }

  const searchKnowledgeBase = async (
    query: string,
    topK: number = 5,
  ): Promise<KnowledgeBaseSearchResponse | null> => {
    if (!isRAGEnabled.value) {
      console.warn('RAG feature is disabled')
      return null
    }
    loading.value = true
    error.value = null
    try {
      const response = await api.rag.searchKnowledgeBase({
        query,
        topK,
        filters: {},
      })
      return response as KnowledgeBaseSearchResponse
    }
    catch (err: any) {
      console.error('Knowledge base search error:', err)
      error.value = err
      return null
    }
    finally {
      loading.value = false
    }
  }

  const explainAnomaly = async (anomalyData: any): Promise<string | null> => {
    if (!isRAGEnabled.value) {
      return null
    }
    loading.value = true
    error.value = null
    try {
      const response = await api.rag.explainAnomaly({
        anomalyData,
        includeRecommendations: true,
      })
      return response?.explanation || null
    }
    catch (err: any) {
      console.error('Anomaly explanation error:', err)
      error.value = err
      return null
    }
    finally {
      loading.value = false
    }
  }

  const clearError = (): void => {
    error.value = null
  }

  const checkHealth = async (): Promise<boolean> => {
    try {
      const response = await fetch(`${config.public.apiBase}/rag/health`)
      return response.ok
    }
    catch {
      return false
    }
  }

  return {
    interpretDiagnosis,
    searchKnowledgeBase,
    explainAnomaly,
    checkHealth,
    clearError,
    loading: readonly(loading),
    error: readonly(error),
    lastInterpretation: readonly(lastInterpretation),
    isRAGEnabled: readonly(isRAGEnabled),
  }
}

export function parseRAGResponse(rawResponse: string): Partial<RAGInterpretationResponse> {
  const sections: Partial<RAGInterpretationResponse> = {}
  try {
    const reasoningMatch = rawResponse.match(/<думает>([\s\S]*?)<\/думает>/i)
    if (reasoningMatch) {
      sections.reasoning = reasoningMatch[1].trim()
    }
    const summaryMatch = rawResponse.match(/<резюме>([\s\S]*?)<\/резюме>/i)
    if (summaryMatch) {
      sections.summary = summaryMatch[1].trim()
    }
    const analysisMatch = rawResponse.match(/<анализ>([\s\S]*?)<\/анализ>/i)
    if (analysisMatch) {
      sections.analysis = analysisMatch[1].trim()
    }
    const recommendationsMatch = rawResponse.match(/<рекомендации>([\s\S]*?)<\/рекомендации>/i)
    if (recommendationsMatch) {
      const recText = recommendationsMatch[1].trim()
      sections.recommendations = recText
        .split(/\n+/)
        .map(line => line.replace(/^\d+\.\s*/, '').trim())
        .filter(line => line.length > 0)
    }
    if (!sections.reasoning && !sections.summary && !sections.analysis) {
      sections.summary = rawResponse.substring(0, 500)
      sections.analysis = rawResponse
    }
  }
  catch (err) {
    console.error('Failed to parse RAG response:', err)
    sections.summary = 'Ошибка парсинга ответа'
    sections.analysis = rawResponse
  }
  return sections
}

export function getConfidenceLevel(confidence: number): {
  level: 'high' | 'medium' | 'low'
  color: string
  label: string
} {
  if (confidence >= 0.8) {
    return { level: 'high', color: 'green', label: 'Высокая' }
  }
  else if (confidence >= 0.5) {
    return { level: 'medium', color: 'yellow', label: 'Средняя' }
  }
  else {
    return { level: 'low', color: 'red', label: 'Низкая' }
  }
}
