<script setup lang="ts">
/**
 * RAGInterpretation - Human-readable diagnosis interpretation
 * 
 * Displays DeepSeek-R1 RAG interpretation with:
 * - Summary with health indicator
 * - Expandable reasoning section
 * - Prioritized recommendations
 * - Prognosis card
 * - Technical details (collapsible)
 * 
 * @example
 * ```vue
 * <RAGInterpretation :interpretation="ragResult" />
 * ```
 */

import type { RAGInterpretation as RAGType } from '~/generated/api'

interface Props {
  interpretation: RAGType
}

const props = defineProps<Props>()

// State
const showReasoning = ref(false)
const showTechnical = ref(false)

// Health score color
const healthScoreColor = computed(() => {
  const score = props.interpretation.health_score || 0
  if (score >= 80) return 'text-green-600'
  if (score >= 60) return 'text-yellow-600'
  return 'text-red-600'
})

// Severity color
function getSeverityColor(severity: string): string {
  const colors: Record<string, string> = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800'
  }
  return colors[severity] || 'bg-gray-100 text-gray-800'
}

// Priority icon
function getPriorityIcon(priority: string): string {
  const icons: Record<string, string> = {
    high: 'heroicons:exclamation-circle',
    medium: 'heroicons:information-circle',
    low: 'heroicons:check-circle'
  }
  return icons[priority] || 'heroicons:information-circle'
}
</script>

<template>
  <div class="rag-interpretation">
    <!-- Summary Card -->
    <div class="summary-card">
      <div class="flex items-start gap-6">
        <!-- Health score circular -->
        <div class="health-indicator">
          <svg class="health-circle" viewBox="0 0 100 100">
            <circle
              class="health-bg"
              cx="50"
              cy="50"
              r="45"
            />
            <circle
              class="health-progress"
              cx="50"
              cy="50"
              r="45"
              :stroke-dasharray="`${(interpretation.health_score || 0) * 2.827} 282.7`"
            />
          </svg>
          <div class="health-value">
            <span :class="['text-3xl font-bold', healthScoreColor]">
              {{ interpretation.health_score || 0 }}
            </span>
            <span class="text-sm text-gray-500">/100</span>
          </div>
        </div>
        
        <!-- Summary text -->
        <div class="flex-1">
          <h3 class="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Интерпретация результатов
          </h3>
          <p class="text-gray-700 dark:text-gray-300 leading-relaxed">
            {{ interpretation.summary }}
          </p>
          
          <!-- Model badge -->
          <div class="flex items-center gap-2 mt-3">
            <Icon name="heroicons:sparkles" class="w-4 h-4 text-purple-600" />
            <span class="text-xs text-gray-500">
              Powered by {{ interpretation.model || 'DeepSeek-R1' }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Reasoning (expandable) -->
    <div class="reasoning-section">
      <button
        class="reasoning-toggle"
        @click="showReasoning = !showReasoning"
      >
        <Icon
          :name="showReasoning ? 'heroicons:chevron-down' : 'heroicons:chevron-right'"
          class="w-5 h-5"
        />
        <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-purple-600" />
        <span class="font-medium">🧠 Процесс анализа</span>
      </button>
      
      <div v-if="showReasoning" class="reasoning-content">
        <pre class="reasoning-text">{{ interpretation.reasoning }}</pre>
      </div>
    </div>

    <!-- Recommendations -->
    <div class="recommendations-section">
      <h3 class="section-title">
        <Icon name="heroicons:light-bulb" class="w-5 h-5 text-yellow-500" />
        Рекомендации
      </h3>
      
      <div class="recommendations-list">
        <div
          v-for="(rec, index) in interpretation.recommendations"
          :key="index"
          class="recommendation-item"
        >
          <div class="flex items-start gap-3">
            <Icon
              :name="getPriorityIcon(rec.priority)"
              :class="['w-5 h-5 flex-shrink-0', 
                rec.priority === 'high' ? 'text-red-600' : 
                rec.priority === 'medium' ? 'text-yellow-600' : 'text-green-600'
              ]"
            />
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ rec.action }}
                </span>
                <span :class="['priority-badge', getSeverityColor(rec.priority)]">
                  {{ rec.priority }}
                </span>
              </div>
              <p class="text-sm text-gray-600 dark:text-gray-400">
                {{ rec.description }}
              </p>
              <div v-if="rec.estimated_time" class="text-xs text-gray-500 mt-1">
                ⛱️ {{ rec.estimated_time }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Prognosis -->
    <div v-if="interpretation.prognosis" class="prognosis-section">
      <h3 class="section-title">
        <Icon name="heroicons:calendar" class="w-5 h-5 text-blue-500" />
        Прогноз
      </h3>
      <p class="text-gray-700 dark:text-gray-300">
        {{ interpretation.prognosis }}
      </p>
    </div>

    <!-- Technical Details (collapsible) -->
    <div class="technical-section">
      <button
        class="technical-toggle"
        @click="showTechnical = !showTechnical"
      >
        <Icon
          :name="showTechnical ? 'heroicons:chevron-down' : 'heroicons:chevron-right'"
          class="w-5 h-5"
        />
        <span>🔧 Технические детали</span>
      </button>
      
      <div v-if="showTechnical" class="technical-content">
        <dl class="technical-list">
          <div>
            <dt>Модель</dt>
            <dd>{{ interpretation.model }}</dd>
          </div>
          <div>
            <dt>Уверенность</dt>
            <dd>{{ (interpretation.confidence * 100).toFixed(1) }}%</dd>
          </div>
          <div>
            <dt>Время обработки</dt>
            <dd>{{ interpretation.processing_time_ms }} ms</dd>
          </div>
        </dl>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rag-interpretation {
  @apply space-y-6;
}

.summary-card {
  @apply bg-gradient-to-br from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-900;
  @apply border border-blue-200 dark:border-gray-700 rounded-lg p-6;
}

.health-indicator {
  @apply relative w-32 h-32 flex-shrink-0;
}

.health-circle {
  @apply w-full h-full transform -rotate-90;
}

.health-bg {
  @apply fill-none stroke-gray-200 dark:stroke-gray-700;
  stroke-width: 8;
}

.health-progress {
  @apply fill-none stroke-blue-600;
  stroke-width: 8;
  stroke-linecap: round;
  transition: stroke-dasharray 1s ease;
}

.health-value {
  @apply absolute inset-0 flex flex-col items-center justify-center;
}

.reasoning-section,
.technical-section {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700;
}

.reasoning-toggle,
.technical-toggle {
  @apply w-full flex items-center gap-2 px-4 py-3;
  @apply text-gray-700 dark:text-gray-300;
  @apply hover:bg-gray-50 dark:hover:bg-gray-700;
  @apply transition-colors;
}

.reasoning-content,
.technical-content {
  @apply px-4 pb-4;
}

.reasoning-text {
  @apply text-sm text-gray-600 dark:text-gray-400;
  @apply bg-gray-50 dark:bg-gray-900 p-4 rounded-lg;
  @apply overflow-x-auto whitespace-pre-wrap;
  @apply font-mono;
}

.recommendations-section,
.prognosis-section {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6;
}

.section-title {
  @apply flex items-center gap-2 text-lg font-semibold text-gray-900 dark:text-white mb-4;
}

.recommendations-list {
  @apply space-y-4;
}

.recommendation-item {
  @apply p-4 bg-gray-50 dark:bg-gray-900 rounded-lg;
}

.priority-badge {
  @apply px-2 py-0.5 rounded-full text-xs font-medium;
}

.technical-list {
  @apply grid grid-cols-2 gap-4;
}

.technical-list dt {
  @apply text-sm text-gray-500;
}

.technical-list dd {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}
</style>
