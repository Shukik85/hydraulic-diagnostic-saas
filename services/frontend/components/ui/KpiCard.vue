<template>
  <div class="card-glass p-6 card-hover">
    <!-- Loading state -->
    <div v-if="isLoading" class="animate-pulse">
      <div class="flex items-center justify-between mb-4">
        <div class="skeleton-text w-24" />
        <div class="skeleton-circle w-12 h-12" />
      </div>
      <div class="skeleton-title w-20 mb-2" />
      <div class="skeleton-text w-16" />
    </div>

    <!-- Content state -->
    <div v-else>
      <!-- Header -->
      <div class="flex items-start justify-between mb-4">
        <div class="flex-1">
          <p class="text-sm text-steel-shine font-medium mb-1">
            {{ title }}
          </p>
          <div class="flex items-baseline gap-2">
            <span class="text-4xl font-bold text-white">
              {{ typeof value === 'number' ? value.toLocaleString() : value }}
            </span>
            <span v-if="unit" class="text-sm text-steel-400">
              {{ unit }}
            </span>
          </div>
        </div>

        <!-- Icon -->
        <div 
          class="w-12 h-12 rounded-lg flex items-center justify-center transition-all duration-300"
          :class="iconBgClass"
        >
          <Icon 
            :name="icon" 
            class="w-6 h-6"
            :class="iconColorClass"
          />
        </div>
      </div>

      <!-- Trend -->
      <div 
        v-if="growth !== undefined"
        class="flex items-center gap-1.5"
        :class="trendColorClass"
      >
        <Icon 
          :name="trendIcon" 
          class="w-4 h-4"
        />
        <span class="text-sm font-medium">
          {{ formatGrowth(growth) }}
        </span>
        <span class="text-xs text-steel-400">
          {{ trendLabel || 'vs прошлый период' }}
        </span>
      </div>

      <!-- Helper Text -->
      <UHelperText 
        v-if="description"
        :text="description"
        class="mt-3"
        :show-icon="false"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  value: string | number
  icon: string
  color?: 'primary' | 'success' | 'warning' | 'error' | 'info' | 'steel'
  growth?: number
  loadingState?: string | boolean
  description?: string
  unit?: string
  trendLabel?: string
}

const props = withDefaults(defineProps<Props>(), {
  color: 'primary',
  loadingState: false,
  description: undefined,
  unit: undefined,
  trendLabel: undefined,
  growth: undefined,
})

type ColorKey = NonNullable<Props['color']>

const iconBgClass = computed(() => {
  const classes: Record<ColorKey, string> = {
    primary: 'bg-primary-600/10',
    success: 'bg-success-600/10',
    warning: 'bg-yellow-600/10',
    error: 'bg-red-600/10',
    info: 'bg-blue-600/10',
    steel: 'bg-steel-600/10',
  }
  return classes[props.color]
})

const iconColorClass = computed(() => {
  const classes: Record<ColorKey, string> = {
    primary: 'text-primary-400',
    success: 'text-success-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
    info: 'text-blue-400',
    steel: 'text-steel-400',
  }
  return classes[props.color]
})

const trendColorClass = computed(() => {
  if (props.growth === undefined || props.growth === null) return 'text-steel-400'
  return props.growth >= 0 ? 'text-success-400' : 'text-red-400'
})

const trendIcon = computed(() => {
  if (props.growth === undefined || props.growth === null) return 'heroicons:minus'
  return props.growth >= 0 ? 'heroicons:arrow-trending-up' : 'heroicons:arrow-trending-down'
})

const formatGrowth = (value?: number): string => {
  if (value === undefined || value === null) return ''
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(1)}%`
}

const isLoading = computed(() => {
  return (
    props.loadingState === true || 
    props.loadingState === 'loading' || 
    props.loadingState === 'true'
  )
})
</script>
