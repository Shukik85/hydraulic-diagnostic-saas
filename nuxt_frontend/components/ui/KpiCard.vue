<script setup lang="ts">
// Professional KPI card component with loading states and animations
interface Props {
  title: string
  value: string | number
  icon: string
  color?: 'blue' | 'green' | 'purple' | 'orange' | 'teal' | 'red' | 'indigo'
  growth?: number
  loadingState?: string | boolean
  description?: string
}

const props = withDefaults(defineProps<Props>(), {
  color: 'blue',
  loadingState: false
})

// Type-safe color mapping
type ColorKey = NonNullable<Props['color']>

const getColorClasses = (color: ColorKey): string => {
  const colorMap: Record<ColorKey, string> = {
    blue: 'from-blue-500 to-blue-600 bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800',
    green: 'from-green-500 to-green-600 bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800',
    purple: 'from-purple-500 to-purple-600 bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/20 dark:text-purple-300 dark:border-purple-800',
    orange: 'from-orange-500 to-orange-600 bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-900/20 dark:text-orange-300 dark:border-orange-800',
    teal: 'from-teal-500 to-teal-600 bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-900/20 dark:text-teal-300 dark:border-teal-800',
    red: 'from-red-500 to-red-600 bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800',
    indigo: 'from-indigo-500 to-indigo-600 bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-900/20 dark:text-indigo-300 dark:border-indigo-800'
  }

  return colorMap[color]
}

const formatGrowth = (value?: number): string => {
  if (value === undefined || value === null) return ''
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(1)}%`
}

const getGrowthColor = (value?: number): string => {
  if (value === undefined || value === null) return 'text-gray-500'
  return value >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
}

const getGrowthIcon = (value?: number): string => {
  if (value === undefined || value === null) return 'heroicons:minus'
  return value >= 0 ? 'heroicons:arrow-trending-up' : 'heroicons:arrow-trending-down'
}

// Loading state management
const isLoading = computed(() => {
  return props.loadingState === true || props.loadingState === 'loading' || props.loadingState === 'true'
})
</script>

<template>
  <div class="premium-card p-6 premium-card-hover group">
    <!-- Loading state -->
    <div v-if="isLoading" class="animate-pulse">
      <div class="flex items-center justify-between mb-4">
        <div class="w-16 h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
        <div class="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-lg"></div>
      </div>
      <div class="w-20 h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
      <div class="w-12 h-3 bg-gray-300 dark:bg-gray-600 rounded"></div>
    </div>
    
    <!-- Content state -->
    <div v-else>
      <!-- Header -->
      <div class="flex items-center justify-between mb-4">
        <h3 class="premium-body-sm text-gray-600 dark:text-gray-300 font-medium">
          {{ title }}
        </h3>
        <div :class="[
          'w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300',
          'group-hover:scale-110 group-hover:rotate-3',
          `bg-gradient-to-br ${getColorClasses(color || 'blue')}`
        ]">
          <Icon :name="icon" class="w-5 h-5 text-white" />
        </div>
      </div>
      
      <!-- Value -->
      <div class="mb-2">
        <div class="text-3xl font-bold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
          {{ typeof value === 'number' ? value.toLocaleString() : value }}
        </div>
      </div>
      
      <!-- Growth indicator -->
      <div v-if="growth !== undefined" class="flex items-center space-x-2">
        <Icon 
          :name="getGrowthIcon(growth)" 
          :class="['w-4 h-4', getGrowthColor(growth)]" 
        />
        <span :class="['text-sm font-medium', getGrowthColor(growth)]">
          {{ formatGrowth(growth) }}
        </span>
        <span class="text-gray-500 dark:text-gray-400 text-sm">vs прошлый период</span>
      </div>
      
      <!-- Description -->
      <div v-if="description" class="mt-3 text-sm text-gray-600 dark:text-gray-300">
        {{ description }}
      </div>
    </div>
  </div>
</template>