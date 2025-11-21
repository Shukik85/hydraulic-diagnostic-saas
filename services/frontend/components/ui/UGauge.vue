<template>
  <div class="flex flex-col items-center">
    <!-- Gauge SVG -->
    <div class="relative w-full max-w-[200px] aspect-square">
      <svg viewBox="0 0 200 200" class="w-full h-full transform -rotate-90">
        <!-- Background Circle -->
        <circle cx="100" cy="100" :r="radius" fill="none" :stroke="bgColor" :stroke-width="strokeWidth"
          class="opacity-20" />

        <!-- Progress Arc -->
        <circle cx="100" cy="100" :r="radius" fill="none" :stroke="gaugeColor" :stroke-width="strokeWidth"
          :stroke-dasharray="circumference" :stroke-dashoffset="dashOffset" stroke-linecap="round"
          class="transition-all duration-1000 ease-out" :class="animated ? 'animate-gauge' : ''" />
      </svg>

      <!-- Center Content -->
      <div class="absolute inset-0 flex flex-col items-center justify-center">
        <div class="text-4xl font-bold text-white">
          {{ displayValue }}
        </div>
        <div class="text-sm text-steel-shine mt-1">
          {{ unit }}
        </div>
      </div>
    </div>

    <!-- Label -->
    <div v-if="label" class="text-center mt-4 text-sm font-medium text-steel-shine">
      {{ label }}
    </div>

    <!-- Status Badge -->
    <UBadge v-if="showStatus" :variant="statusVariant" class="mt-2">
      {{ statusText }}
    </UBadge>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  value: number
  max: number
  min?: number
  unit?: string
  label?: string
  size?: number
  strokeWidth?: number
  color?: string
  bgColor?: string
  showStatus?: boolean
  statusThresholds?: {
    success: number
    warning: number
  }
  animated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  min: 0,
  unit: '%',
  label: undefined,
  size: 200,
  strokeWidth: 12,
  color: undefined,
  bgColor: '#64748b',
  showStatus: false,
  statusThresholds: () => ({ success: 80, warning: 50 }),
  animated: true,
})

const radius = computed(() => (props.size - props.strokeWidth) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)

const percentage = computed(() => {
  const range = props.max - props.min
  const normalizedValue = props.value - props.min
  return Math.min(Math.max((normalizedValue / range) * 100, 0), 100)
})

const dashOffset = computed(() => {
  const progress = percentage.value / 100
  return circumference.value * (1 - progress)
})

const displayValue = computed(() => {
  if (props.unit === '%') {
    return Math.round(percentage.value)
  }
  return Math.round(props.value)
})

const gaugeColor = computed(() => {
  if (props.color) return props.color

  const percent = percentage.value
  if (percent >= props.statusThresholds.success) return '#10b981' // success
  if (percent >= props.statusThresholds.warning) return '#f59e0b' // warning
  return '#ef4444' // error
})

const statusVariant = computed(() => {
  const percent = percentage.value
  if (percent >= props.statusThresholds.success) return 'success'
  if (percent >= props.statusThresholds.warning) return 'warning'
  return 'destructive'
})

const statusText = computed(() => {
  const percent = percentage.value
  if (percent >= props.statusThresholds.success) return 'Отлично'
  if (percent >= props.statusThresholds.warning) return 'Нормально'
  return 'Низкий'
})
</script>

<style scoped>
@keyframes gauge {
  from {
    stroke-dashoffset: v-bind(circumference);
  }
}

.animate-gauge {
  animation: gauge 1.5s ease-out;
}
</style>
