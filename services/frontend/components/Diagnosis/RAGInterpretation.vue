<template>
  <div class="space-y-6">
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
            {{ error }}
          </p>
        </div>
      </div>
    </div>

    <!-- Interpretation Content -->
    <div v-else-if="interpretation" class="space-y-6">
      <!-- Summary -->
      <div class="u-card p-6 border-l-4 border-blue-500 bg-blue-50">
        <h4 class="font-semibold text-blue-900 mb-2">
          📊 Краткая сводка
        </h4>
        <p class="u-body text-blue-800 whitespace-pre-wrap">
          {{ interpretation.summary }}
        </p>
      </div>

      <!-- Reasoning -->
      <div v-if="interpretation.reasoning" class="u-card">
        <details class="group">
          <summary class="cursor-pointer p-6 flex items-center justify-between hover:bg-gray-50 transition">
            <h4 class="font-semibold text-gray-900">
              🧠 Процесс рассуждения
            </h4>
            <svg class="w-5 h-5 text-gray-400 group-open:rotate-180 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </summary>
          <div class="p-6 pt-0 border-t">
            <ReasoningViewer :reasoning="interpretation.reasoning" />
          </div>
        </details>
      </div>

      <!-- Recommendations -->
      <div v-if="interpretation.recommendations?.length" class="u-card p-6">
        <h4 class="font-semibold text-gray-900 mb-4">
          💡 Рекомендации
        </h4>
        <div class="space-y-3">
          <div v-for="(rec, index) in interpretation.recommendations" :key="index" class="flex items-start gap-3 p-4 bg-orange-50 border border-orange-200 rounded-lg">
            <div class="shrink-0 w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
              {{ index + 1 }}
            </div>
            <p class="text-gray-700 flex-1">
              {{ rec }}
            </p>
          </div>
        </div>
      </div>

      <!-- Confidence -->
      <div class="u-card p-4">
        <div class="text-sm text-gray-600 mb-1">Уверенность</div>
        <div class="text-lg font-semibold" :class="{
          'text-green-600': interpretation.confidence >= 0.8,
          'text-yellow-600': interpretation.confidence >= 0.5,
          'text-red-600': interpretation.confidence < 0.5
        }">
          {{ Math.round(interpretation.confidence * 100) }}%
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from '#imports'
import type { RAGInterpretationResponse } from '~/types/rag'
import ReasoningViewer from './ReasoningViewer.vue'

interface Props {
  interpretation: RAGInterpretationResponse | null
  loading?: boolean
  error?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  error: null,
})
</script>
