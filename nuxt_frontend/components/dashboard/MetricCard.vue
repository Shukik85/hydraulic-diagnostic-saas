<template>
  <div class="premium-card p-6">
    <div class="flex items-center">
      <div class="flex-shrink-0">
        <div 
          class="w-10 h-10 rounded-full flex items-center justify-center premium-transition"
          :class="iconBgClass"
        >
          <Icon 
            :name="icon" 
            class="w-5 h-5"
            :class="iconClass"
          />
        </div>
      </div>
      
      <div class="ml-4 flex-1">
        <div class="premium-body text-gray-600 dark:text-gray-400">{{ title }}</div>
        <div class="premium-heading-lg text-gray-900 dark:text-white">
          {{ formatValue(value) }}
          <span v-if="unit" class="premium-body text-gray-500 dark:text-gray-400 ml-1">{{ unit }}</span>
        </div>
        
        <div v-if="trend" class="flex items-center mt-1">
          <Icon 
            :name="trendIcon" 
            class="w-4 h-4 mr-1"
            :class="trendClass"
          />
          <span class="text-xs font-medium" :class="trendClass">
            {{ trend }}
          </span>
        </div>
      </div>
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
    success: 'bg-green-100 dark:bg-green-900/20',
    warning: 'bg-yellow-100 dark:bg-yellow-900/20', 
    error: 'bg-red-100 dark:bg-red-900/20',
    info: 'bg-blue-100 dark:bg-blue-900/20',
    default: 'bg-gray-100 dark:bg-gray-700'
  }
  return classes[props.type]
})

const iconClass = computed(() => {
  const classes = {
    success: 'text-status-success',
    warning: 'text-status-warning',
    error: 'text-status-error', 
    info: 'text-blue-600 dark:text-blue-400',
    default: 'text-gray-600 dark:text-gray-400'
  }
  return classes[props.type]
})

const trendIcon = computed(() => {
  if (!props.trend) return ''
  
  if (props.trend.startsWith('+') || !props.trend.startsWith('-')) {
    return 'heroicons:arrow-trending-up'
  } else {
    return 'heroicons:arrow-trending-down'
  }
})

const trendClass = computed(() => {
  if (!props.trend) return ''
  
  if (props.trend.startsWith('+') || !props.trend.startsWith('-')) {
    return 'text-status-success'
  } else {
    return 'text-status-error'
  }
})
</script>