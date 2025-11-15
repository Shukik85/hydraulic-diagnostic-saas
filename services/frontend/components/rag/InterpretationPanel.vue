<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div class="flex-1">
        <h3 class="u-h4 mb-2">
          <span class="mr-2">🤖</span>
          AI-Интерпретация
        </h3>
        <p class="u-body-sm text-gray-600">
          Анализ с помощью DeepSeek-R1 (70B) + Knowledge Base
        </p>
      </div>
      
      <!-- Confidence Badge -->
      <div v-if="interpretation && !loading" class="shrink-0">
        <div class="flex items-center gap-2 px-3 py-2 rounded-lg"
          :class="{
            'bg-green-50 text-green-700': confidenceLevel.level === 'high',
            'bg-yellow-50 text-yellow-700': confidenceLevel.level === 'medium',
            'bg-red-50 text-red-700': confidenceLevel.level === 'low'
          }">
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
          </svg>
          <span class="text-sm font-medium">
            Уверенность: {{ confidenceLevel.label }}
          </span>
          <span class="text-xs opacity-75">
            {{ Math.round(interpretation.confidence * 100) }}%
          </span>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="u-card p-8 text-center">
      <div class="inline-flex items-center gap-3">
        <svg class="animate-spin h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span class="text-gray-600">
          Генерирую интерпретацию...
        </span>
      </div>
      <p class="u-body-sm text-gray-500 mt-2">
        Это может занять 10-30 секунд
      </p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="u-card p-6 border-l-4 border-red-500 bg-red-50">
      <div class="flex items-start gap-3">
        <svg class="w-5 h-5 text-red-600 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
        </svg>
        <div class="flex-1">
          <h4 class="font-medium text-red-800 mb-1">
            Ошибка генерации
          </h4>
          <p class="u-body-sm text-red-700">
            {{ error.message }}
          </p>
          <button @click="$emit('retry')" class="u-btn u-btn-sm u-btn-secondary mt-3">
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Повторить
          </button>
        </div>
      </div>
    </div>

    <!-- Interpretation Content -->
    <div v-else-if="interpretation" class="space-y-6">
      <!-- Summary Card -->
      <div class="u-card p-6 border-l-4 border-blue-500 bg-blue-50">
        <div class="flex items-start gap-3">
          <svg class="w-5 h-5 text-blue-600 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
            <path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd" />
          </svg>
          <div class="flex-1 min-w-0">
            <h4 class="font-semibold text-blue-900 mb-2">
              📊 Краткая сводка
            </h4>
            <p class="u-body text-blue-800 whitespace-pre-wrap">
              {{ interpretation.summary }}
            </p>
          </div>
        </div>
      </div>

      <!-- Reasoning Process (collapsible) -->
      <div v-if="interpretation.reasoning" class="u-card">
        <details class="group">
          <summary class="cursor-pointer p-6 flex items-center justify-between hover:bg-gray-50 transition">
            <div class="flex items-center gap-3">
              <svg class="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" />
              </svg>
              <h4 class="font-semibold text-gray-900">
                🧠 Процесс рассуждения (Reasoning)
              </h4>
            </div>
            <svg class="w-5 h-5 text-gray-400 group-open:rotate-180 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </summary>
          
          <div class="p-6 pt-0 border-t">
            <div class="bg-gray-50 p-4 rounded-lg">
              <pre class="text-sm text-gray-700 whitespace-pre-wrap font-mono">{{ interpretation.reasoning }}</pre>
            </div>
            <p class="u-body-sm text-gray-500 mt-3">
              ⚡ Это внутренний монолог AI модели - показывает, как она пришла к выводу
            </p>
          </div>
        </details>
      </div>

      <!-- Detailed Analysis -->
      <div class="u-card p-6">
        <h4 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
            <path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm9.707 5.707a1 1 0 00-1.414-1.414L9 12.586l-1.293-1.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
          </svg>
          🔍 Детальный анализ
        </h4>
        <div class="prose prose-sm max-w-none">
          <p class="text-gray-700 whitespace-pre-wrap leading-relaxed">
            {{ interpretation.analysis }}
          </p>
        </div>
      </div>

      <!-- Recommendations -->
      <div v-if="interpretation.recommendations?.length" class="u-card p-6">
        <h4 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-orange-600" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
          </svg>
          💡 Рекомендации
        </h4>
        <div class="space-y-3">
          <div v-for="(rec, index) in interpretation.recommendations" :key="index"
            class="flex items-start gap-3 p-4 bg-orange-50 border border-orange-200 rounded-lg">
            <div class="shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
              {{ index + 1 }}
            </div>
            <p class="text-gray-700 flex-1">
              {{ rec }}
            </p>
          </div>
        </div>
      </div>

      <!-- Knowledge Base Context -->
      <div v-if="interpretation.knowledgeUsed?.length" class="u-card p-6 bg-gradient-to-br from-purple-50 to-blue-50">
        <h4 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
          </svg>
          📚 Использованные документы
        </h4>
        <p class="u-body-sm text-gray-600 mb-4">
          AI использовала эти документы из базы знаний:
        </p>
        <div class="space-y-2">
          <div v-for="doc in interpretation.knowledgeUsed" :key="doc.id"
            class="p-3 bg-white rounded-lg border border-purple-200 hover:border-purple-400 transition cursor-pointer">
            <div class="flex items-center justify-between gap-3 mb-2">
              <h5 class="font-medium text-gray-900 text-sm">
                {{ doc.title }}
              </h5>
              <span class="text-xs font-medium text-purple-600">
                {{ Math.round(doc.score * 100) }}% match
              </span>
            </div>
            <p class="u-body-sm text-gray-600 line-clamp-2">
              {{ doc.content }}
            </p>
          </div>
        </div>
      </div>

      <!-- Metadata -->
      <div v-if="interpretation.metadata" class="text-center">
        <div class="inline-flex items-center gap-4 text-xs text-gray-500">
          <span>
            <svg class="inline w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd" />
            </svg>
            {{ (interpretation.metadata.processingTime / 1000).toFixed(1) }}s
          </span>
          <span>
            <svg class="inline w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 0l-2 2a1 1 0 101.414 1.414L8 10.414l1.293 1.293a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
            </svg>
            {{ interpretation.metadata.tokensUsed.toLocaleString() }} tokens
          </span>
          <span>
            {{ interpretation.metadata.model }}
          </span>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="u-card p-12 text-center">
      <div class="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
        <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      </div>
      <h4 class="u-h5 text-gray-900 mb-2">
        Интерпретация не запущена
      </h4>
      <p class="u-body text-gray-600 mb-6">
        Нажмите "Генерировать", чтобы получить AI-анализ результатов диагностики
      </p>
      <button @click="$emit('generate')" class="u-btn u-btn-primary">
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Генерировать интерпретацию
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { RAGInterpretationResponse } from '~/types/rag'
import { getConfidenceLevel } from '~/composables/useRAG'

interface Props {
  interpretation: RAGInterpretationResponse | null
  loading?: boolean
  error?: Error | null
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  error: null
})

defineEmits<{
  retry: []
  generate: []
}>()

const confidenceLevel = computed(() => {
  if (!props.interpretation) {
    return { level: 'low' as const, color: 'gray', label: 'Н/Д' }
  }
  return getConfidenceLevel(props.interpretation.confidence)
})
</script>