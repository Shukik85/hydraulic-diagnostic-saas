<template>
  <div class="u-metric-card">
    <div class="u-metric-header">
      <h3 class="u-metric-label">{{ title }}</h3>
      <div class="u-metric-icon" :class="iconBgClass">
        <Icon 
          :name="icon" 
          class="w-5 h-5"
          :class="iconClass"
        />
      </div>
    </div>
    
    <div class="u-metric-value">
      {{ formatValue(value) }}
      <span v-if="unit" class="text-lg text-gray-500 dark:text-gray-400 ml-1">{{ unit }}</span>
    </div>
    
    <div v-if="trend" class="u-metric-change mt-2" :class="trendClass">
      <Icon 
        :name="trendIcon" 
        class="w-4 h-4"
      />
      <span>{{ trend }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  value: number | string
  unit?: string
  trend?: string
  icon?: string
  type?: 'success' | 'warning' | 'error' | 'info' | 'default'
}

const props = withDefaults(defineProps<Props>(), {
  icon: 'heroicons:chart-bar',
  type: 'default'
})

const formatValue = (val: number | string) => {
  if (typeof val === 'number') {
    return val.toLocaleString('ru-RU', { maximumFractionDigits: 1 })
  }
  return val
}

const iconBgClass = computed(() => {
  const classes = {
    success: 'bg-green-100 dark:bg-green-900/30',
    warning: 'bg-yellow-100 dark:bg-yellow-900/30', 
    error: 'bg-red-100 dark:bg-red-900/30',
    info: 'bg-blue-100 dark:bg-blue-900/30',
    default: 'bg-gray-100 dark:bg-gray-700'
  }
  return classes[props.type]
})

const iconClass = computed(() => {
  const classes = {
    success: 'text-green-600 dark:text-green-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    error: 'text-red-600 dark:text-red-400', 
    info: 'text-blue-600 dark:text-blue-400',
    default: 'text-gray-600 dark:text-gray-400'
  }
  return classes[props.type]
})

const trendIcon = computed(() => {
  if (!props.trend) return ''
  
  if (props.trend.includes('+') || props.trend.includes('up') || props.trend.includes('improvement')) {
    return 'heroicons:arrow-trending-up'
  } else if (props.trend.includes('-') || props.trend.includes('down')) {
    return 'heroicons:arrow-trending-down'
  }
  return 'heroicons:minus'
})

const trendClass = computed(() => {
  if (!props.trend) return ''
  
  if (props.trend.includes('+') || props.trend.includes('up') || props.trend.includes('improvement')) {
    return 'u-metric-change-positive'
  } else if (props.trend.includes('-') || props.trend.includes('down')) {
    return 'u-metric-change-negative'
  }
  return 'text-gray-600 dark:text-gray-400'
})
</script>