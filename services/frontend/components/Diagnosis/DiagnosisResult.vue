<template>
  <div class="diagnosis-result" role="region" aria-labelledby="diagnosis-result-title">
    <!-- Header -->
    <div class="mb-6">
      <h2 id="diagnosis-result-title" class="u-h3 mb-2">
        🔍 Результаты диагностики
      </h2>
      <p class="u-body-sm text-gray-600">
        Оборудование: <strong>{{ equipmentName }}</strong> | Время: {{ formattedTimestamp }}
      </p>
    </div>

    <!-- Overall Status -->
    <div class="u-card p-6 mb-6" :class="statusColorClass">
      <div class="flex items-center gap-4">
        <div class="shrink-0">
          <div class="w-16 h-16 rounded-full flex items-center justify-center" :class="statusBgClass">
            <component :is="statusIcon" class="w-8 h-8" :class="statusIconClass" />
          </div>
        </div>
        <div class="flex-1">
          <div class="text-sm font-medium text-gray-600 mb-1">Общий статус</div>
          <div class="text-2xl font-bold" :class="statusTextClass">
            {{ overallStatus }}
          </div>
        </div>
        <div class="text-right">
          <div class="text-sm text-gray-600 mb-1">Уровень уверенности</div>
          <div class="text-3xl font-bold" :class="confidenceColorClass">
            {{ Math.round(confidence * 100) }}%
          </div>
        </div>
      </div>
    </div>

    <!-- Predictions Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div v-for="pred in predictions" :key="pred.model" class="u-card p-4">
        <div class="text-xs font-medium text-gray-500 mb-2">
          {{ pred.model }}
        </div>
        <div class="text-lg font-semibold mb-1" :class="getPredictionColor(pred.prediction)">
          {{ pred.prediction }}
        </div>
        <div class="text-sm text-gray-600">
          {{ Math.round(pred.confidence * 100) }}% уверенности
        </div>
      </div>
    </div>

    <!-- Feature Importance -->
    <div v-if="featureImportance.length" class="u-card p-6 mb-6">
      <h3 class="u-h5 mb-4">
        📈 Важность признаков
      </h3>
      <div class="space-y-3">
        <div v-for="(feature, idx) in featureImportance.slice(0, 5)" :key="idx" class="flex items-center gap-3">
          <div class="w-32 text-sm font-medium text-gray-700">
            {{ feature.name }}
          </div>
          <div class="flex-1">
            <div class="h-6 bg-gray-200 rounded-full overflow-hidden">
              <div class="h-full bg-blue-500 transition-all" :style="{ width: `${feature.importance * 100}%` }"></div>
            </div>
          </div>
          <div class="w-16 text-right text-sm text-gray-600">
            {{ Math.round(feature.importance * 100) }}%
          </div>
        </div>
      </div>
    </div>

    <!-- Anomalies Detected -->
    <div v-if="anomalies.length" class="u-card p-6 border-l-4 border-red-500">
      <h3 class="u-h5 mb-4 text-red-800">
        ⚠️ Обнаруженные аномалии
      </h3>
      <div class="space-y-3">
        <div v-for="(anomaly, idx) in anomalies" :key="idx" class="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div class="flex items-start justify-between gap-4 mb-2">
            <div class="font-semibold text-red-900">
              {{ anomaly.parameter }}
            </div>
            <div class="text-sm font-medium text-red-700">
              Серьёзность: {{ anomaly.severity }}
            </div>
          </div>
          <div class="text-sm text-red-800 mb-2">
            {{ anomaly.description }}
          </div>
          <div class="flex items-center gap-4 text-xs text-red-700">
            <span>Значение: {{ anomaly.value }}</span>
            <span>Норма: {{ anomaly.expectedRange }}</span>
            <span>Отклонение: {{ Math.round(anomaly.deviation * 100) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from '#imports'
import type { DiagnosticResult, ModelPrediction, Anomaly, FeatureImportance } from '~/types/diagnostics'

interface Props {
  result: DiagnosticResult
  equipmentName?: string
}

const props = withDefaults(defineProps<Props>(), {
  equipmentName: 'Неизвестно',
})

const predictions = computed<ModelPrediction[]>(() => props.result.predictions || [])
const anomalies = computed<Anomaly[]>(() => props.result.anomalies || [])
const featureImportance = computed<FeatureImportance[]>(() => props.result.featureImportance || [])
const confidence = computed<number>(() => props.result.confidence || 0)

const overallStatus = computed<string>(() => {
  const status = props.result.status || 'unknown'
  const statusMap: Record<string, string> = {
    normal: 'Нормальное состояние',
    warning: 'Предупреждение',
    critical: 'Критическое',
    unknown: 'Неизвестно',
  }
  return statusMap[status] || status
})

const formattedTimestamp = computed<string>(() => {
  if (!props.result.timestamp) return 'Н/Д'
  return new Date(props.result.timestamp).toLocaleString('ru-RU')
})

const statusColorClass = computed<string>(() => {
  const status = props.result.status || 'unknown'
  const colorMap: Record<string, string> = {
    normal: 'border-l-4 border-green-500 bg-green-50',
    warning: 'border-l-4 border-yellow-500 bg-yellow-50',
    critical: 'border-l-4 border-red-500 bg-red-50',
    unknown: 'border-l-4 border-gray-500 bg-gray-50',
  }
  return colorMap[status] || colorMap.unknown!
})

const statusBgClass = computed<string>(() => {
  const status = props.result.status || 'unknown'
  const bgMap: Record<string, string> = {
    normal: 'bg-green-100',
    warning: 'bg-yellow-100',
    critical: 'bg-red-100',
    unknown: 'bg-gray-100',
  }
  return bgMap[status] || bgMap.unknown!
})

const statusIconClass = computed<string>(() => {
  const status = props.result.status || 'unknown'
  const iconColorMap: Record<string, string> = {
    normal: 'text-green-600',
    warning: 'text-yellow-600',
    critical: 'text-red-600',
    unknown: 'text-gray-600',
  }
  return iconColorMap[status] || iconColorMap.unknown!
})

const statusTextClass = computed<string>(() => {
  const status = props.result.status || 'unknown'
  const textColorMap: Record<string, string> = {
    normal: 'text-green-700',
    warning: 'text-yellow-700',
    critical: 'text-red-700',
    unknown: 'text-gray-700',
  }
  return textColorMap[status] || textColorMap.unknown!
})

const statusIcon = computed(() => {
  const status = props.result.status || 'unknown'
  // Возвращаем inline SVG или компонент иконки
  const icons: Record<string, any> = {
    normal: 'svg', // CheckCircle icon
    warning: 'svg', // ExclamationCircle icon
    critical: 'svg', // XCircle icon
    unknown: 'svg', // QuestionMarkCircle icon
  }
  return icons[status] || icons.unknown
})

const confidenceColorClass = computed<string>(() => {
  if (confidence.value >= 0.8) return 'text-green-600'
  if (confidence.value >= 0.5) return 'text-yellow-600'
  return 'text-red-600'
})

const getPredictionColor = (prediction: string): string => {
  const predMap: Record<string, string> = {
    normal: 'text-green-600',
    anomaly: 'text-red-600',
    warning: 'text-yellow-600',
  }
  return predMap[prediction.toLowerCase()] || 'text-gray-600'
}
</script>

<style scoped>
.diagnosis-result {
  @apply max-w-7xl mx-auto;
}
</style>
