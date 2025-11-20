<template>
  <div class="diagnosis-comparison" role="region" aria-labelledby="comparison-title">
    <div class="u-card p-6">
      <h2 id="comparison-title" class="u-h4 mb-6">
        🔀 Сравнение диагностик
      </h2>

      <!-- Header with dates -->
      <div class="grid grid-cols-2 gap-6 mb-6">
        <div class="text-center">
          <div class="text-sm text-gray-600 mb-1">Базовая диагностика</div>
          <div class="text-lg font-semibold text-gray-900">
            {{ formatDate(baseline.timestamp) }}
          </div>
        </div>
        <div class="text-center">
          <div class="text-sm text-gray-600 mb-1">Сравниваемая диагностика</div>
          <div class="text-lg font-semibold text-gray-900">
            {{ formatDate(comparison.timestamp) }}
          </div>
        </div>
      </div>

      <!-- Status Comparison -->
      <div class="mb-8">
        <h3 class="u-h6 mb-4">Общий статус</h3>
        <div class="grid grid-cols-2 gap-6">
          <div class="u-card p-4" :class="getStatusBorderClass(baseline.status)">
            <div class="text-center">
              <div class="text-2xl font-bold mb-2" :class="getStatusTextClass(baseline.status)">
                {{ getStatusLabel(baseline.status) }}
              </div>
              <div class="text-sm text-gray-600">
                Уверенность: {{ Math.round(baseline.confidence * 100) }}%
              </div>
            </div>
          </div>
          <div class="u-card p-4" :class="getStatusBorderClass(comparison.status)">
            <div class="text-center">
              <div class="text-2xl font-bold mb-2" :class="getStatusTextClass(comparison.status)">
                {{ getStatusLabel(comparison.status) }}
              </div>
              <div class="text-sm text-gray-600">
                Уверенность: {{ Math.round(comparison.confidence * 100) }}%
              </div>
              <!-- Trend Indicator -->
              <div v-if="statusTrend" class="mt-3 flex items-center justify-center gap-2">
                <svg v-if="statusTrend === 'improved'" class="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clip-rule="evenodd" />
                </svg>
                <svg v-else-if="statusTrend === 'degraded'" class="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clip-rule="evenodd" />
                </svg>
                <svg v-else class="w-5 h-5 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v4a1 1 0 102 0V7z" clip-rule="evenodd" />
                </svg>
                <span class="text-sm font-medium" :class="{
                  'text-green-600': statusTrend === 'improved',
                  'text-red-600': statusTrend === 'degraded',
                  'text-gray-600': statusTrend === 'unchanged'
                }">
                  {{ statusTrendLabel }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Model Predictions Comparison -->
      <div v-if="baselinePredictions.length || comparisonPredictions.length" class="mb-8">
        <h3 class="u-h6 mb-4">Предсказания моделей</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-3 text-left font-semibold text-gray-700">Модель</th>
                <th class="px-4 py-3 text-left font-semibold text-gray-700">Базовая</th>
                <th class="px-4 py-3 text-left font-semibold text-gray-700">Сравниваемая</th>
                <th class="px-4 py-3 text-left font-semibold text-gray-700">Изменение</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200">
              <tr v-for="model in modelNames" :key="model" class="hover:bg-gray-50">
                <td class="px-4 py-3 font-medium text-gray-900">{{ model }}</td>
                <td class="px-4 py-3">
                  <div class="flex items-center gap-2">
                    <span :class="getPredictionColor(getModelPrediction(baselinePredictions, model))">
                      {{ getModelPrediction(baselinePredictions, model) || 'N/A' }}
                    </span>
                    <span class="text-xs text-gray-500">
                      ({{ getModelConfidence(baselinePredictions, model) }}%)
                    </span>
                  </div>
                </td>
                <td class="px-4 py-3">
                  <div class="flex items-center gap-2">
                    <span :class="getPredictionColor(getModelPrediction(comparisonPredictions, model))">
                      {{ getModelPrediction(comparisonPredictions, model) || 'N/A' }}
                    </span>
                    <span class="text-xs text-gray-500">
                      ({{ getModelConfidence(comparisonPredictions, model) }}%)
                    </span>
                  </div>
                </td>
                <td class="px-4 py-3">
                  <span v-if="getModelChange(model)" class="px-2 py-1 rounded text-xs font-medium" :class="getChangeClass(model)">
                    {{ getModelChange(model) }}
                  </span>
                  <span v-else class="text-gray-400 text-xs">Без изменений</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Anomalies Comparison -->
      <div v-if="baselineAnomalies.length || comparisonAnomalies.length" class="mb-8">
        <h3 class="u-h6 mb-4">Аномалии</h3>
        <div class="grid grid-cols-2 gap-6">
          <div>
            <div class="text-sm font-medium text-gray-700 mb-3">
              Базовая: {{ baselineAnomalies.length }} аномалий
            </div>
            <div class="space-y-2">
              <div v-for="anomaly in baselineAnomalies.slice(0, 3)" :key="anomaly.parameter" class="p-3 bg-red-50 border border-red-200 rounded text-sm">
                <div class="font-medium text-red-900">{{ anomaly.parameter }}</div>
                <div class="text-red-700 text-xs mt-1">{{ anomaly.description }}</div>
              </div>
            </div>
          </div>
          <div>
            <div class="text-sm font-medium text-gray-700 mb-3">
              Сравниваемая: {{ comparisonAnomalies.length }} аномалий
            </div>
            <div class="space-y-2">
              <div v-for="anomaly in comparisonAnomalies.slice(0, 3)" :key="anomaly.parameter" class="p-3 bg-red-50 border border-red-200 rounded text-sm">
                <div class="font-medium text-red-900">{{ anomaly.parameter }}</div>
                <div class="text-red-700 text-xs mt-1">{{ anomaly.description }}</div>
              </div>
            </div>
          </div>
        </div>
        <!-- Anomaly Trend -->
        <div v-if="anomalyTrend" class="mt-4 p-4 rounded-lg" :class="{
          'bg-green-50 border border-green-200': anomalyTrend === 'decreased',
          'bg-red-50 border border-red-200': anomalyTrend === 'increased',
          'bg-gray-50 border border-gray-200': anomalyTrend === 'unchanged'
        }">
          <div class="flex items-center gap-2">
            <svg v-if="anomalyTrend === 'decreased'" class="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clip-rule="evenodd" />
            </svg>
            <svg v-else-if="anomalyTrend === 'increased'" class="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clip-rule="evenodd" />
            </svg>
            <span class="text-sm font-medium" :class="{
              'text-green-700': anomalyTrend === 'decreased',
              'text-red-700': anomalyTrend === 'increased',
              'text-gray-700': anomalyTrend === 'unchanged'
            }">
              {{ anomalyTrendLabel }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from '#imports'
import type { DiagnosticResult, ModelPrediction, Anomaly } from '~/types/diagnostics'

interface Props {
  baseline: DiagnosticResult
  comparison: DiagnosticResult
}

const props = defineProps<Props>()

const baselinePredictions = computed<ModelPrediction[]>(() => props.baseline.predictions || [])
const comparisonPredictions = computed<ModelPrediction[]>(() => props.comparison.predictions || [])
const baselineAnomalies = computed<Anomaly[]>(() => props.baseline.anomalies || [])
const comparisonAnomalies = computed<Anomaly[]>(() => props.comparison.anomalies || [])

const modelNames = computed<string[]>(() => {
  const names = new Set<string>()
  baselinePredictions.value.forEach(p => names.add(p.model))
  comparisonPredictions.value.forEach(p => names.add(p.model))
  return Array.from(names)
})

const statusTrend = computed<'improved' | 'degraded' | 'unchanged' | null>(() => {
  const statusOrder = { normal: 0, warning: 1, critical: 2 }
  const baselineOrder = statusOrder[props.baseline.status as keyof typeof statusOrder] ?? 3
  const comparisonOrder = statusOrder[props.comparison.status as keyof typeof statusOrder] ?? 3

  if (comparisonOrder < baselineOrder) return 'improved'
  if (comparisonOrder > baselineOrder) return 'degraded'
  return 'unchanged'
})

const statusTrendLabel = computed<string>(() => {
  if (statusTrend.value === 'improved') return 'Улучшение'
  if (statusTrend.value === 'degraded') return 'Ухудшение'
  return 'Без изменений'
})

const anomalyTrend = computed<'increased' | 'decreased' | 'unchanged' | null>(() => {
  const baseCount = baselineAnomalies.value.length
  const compCount = comparisonAnomalies.value.length

  if (compCount < baseCount) return 'decreased'
  if (compCount > baseCount) return 'increased'
  return 'unchanged'
})

const anomalyTrendLabel = computed<string>(() => {
  const diff = comparisonAnomalies.value.length - baselineAnomalies.value.length
  if (diff > 0) return `+${diff} аномалий`
  if (diff < 0) return `${diff} аномалий`
  return 'Без изменений'
})

const formatDate = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString('ru-RU', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const getStatusLabel = (status: string): string => {
  const labels: Record<string, string> = {
    normal: 'Нормально',
    warning: 'Предупреждение',
    critical: 'Критическое',
    unknown: 'Неизвестно',
  }
  return labels[status] || status
}

const getStatusBorderClass = (status: string): string => {
  const classes: Record<string, string> = {
    normal: 'border-l-4 border-green-500 bg-green-50',
    warning: 'border-l-4 border-yellow-500 bg-yellow-50',
    critical: 'border-l-4 border-red-500 bg-red-50',
    unknown: 'border-l-4 border-gray-500 bg-gray-50',
  }
  return classes[status] ?? classes.unknown!
}

const getStatusTextClass = (status: string): string => {
  const classes: Record<string, string> = {
    normal: 'text-green-700',
    warning: 'text-yellow-700',
    critical: 'text-red-700',
    unknown: 'text-gray-700',
  }
  return classes[status] ?? classes.unknown!
}

const getPredictionColor = (prediction: string): string => {
  const colorMap: Record<string, string> = {
    normal: 'text-green-600',
    anomaly: 'text-red-600',
    warning: 'text-yellow-600',
  }
  return colorMap[prediction?.toLowerCase()] || 'text-gray-600'
}

const getModelPrediction = (predictions: ModelPrediction[], modelName: string): string => {
  const pred = predictions.find(p => p.model === modelName)
  return pred?.prediction || ''
}

const getModelConfidence = (predictions: ModelPrediction[], modelName: string): number => {
  const pred = predictions.find(p => p.model === modelName)
  return pred ? Math.round(pred.confidence * 100) : 0
}

const getModelChange = (modelName: string): string => {
  const basePred = getModelPrediction(baselinePredictions.value, modelName)
  const compPred = getModelPrediction(comparisonPredictions.value, modelName)

  if (!basePred || !compPred || basePred === compPred) return ''

  return `${basePred} → ${compPred}`
}

const getChangeClass = (modelName: string): string => {
  const basePred = getModelPrediction(baselinePredictions.value, modelName).toLowerCase()
  const compPred = getModelPrediction(comparisonPredictions.value, modelName).toLowerCase()

  const predOrder = { normal: 0, warning: 1, anomaly: 2 }
  const baseOrder = predOrder[basePred as keyof typeof predOrder] ?? 3
  const compOrder = predOrder[compPred as keyof typeof predOrder] ?? 3

  if (compOrder < baseOrder) return 'bg-green-100 text-green-700'
  if (compOrder > baseOrder) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-700'
}
</script>

<style scoped>
.diagnosis-comparison {
  @apply max-w-7xl mx-auto;
}
</style>
