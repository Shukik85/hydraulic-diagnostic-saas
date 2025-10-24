<script setup lang="ts">
interface Props {
  title: string
  value: string | number
  growth?: number
  icon: string
  color?: 'blue' | 'green' | 'purple' | 'orange' | 'teal' | 'red' | 'indigo'
  subtitle?: string
  loading?: boolean
}

withDefaults(defineProps<Props>(), {
  color: 'blue',
  growth: 0,
  loading: false
})

const getColorClasses = (color: string) => {
  const colorMap = {
    blue: 'from-blue-500 to-blue-600 bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800',
    green: 'from-green-500 to-green-600 bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800',
    purple: 'from-purple-500 to-purple-600 bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/20 dark:text-purple-300 dark:border-purple-800',
    orange: 'from-orange-500 to-orange-600 bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-900/20 dark:text-orange-300 dark:border-orange-800',
    teal: 'from-teal-500 to-teal-600 bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-900/20 dark:text-teal-300 dark:border-teal-800',
    red: 'from-red-500 to-red-600 bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800',
    indigo: 'from-indigo-500 to-indigo-600 bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-900/20 dark:text-indigo-300 dark:border-indigo-800'
  }
  return colorMap[color] || colorMap.blue
}

const formatGrowth = (value: number) => {
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(1)}%`
}
</script>

<template>
  <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-105 transition-all duration-300 group">
    <!-- Loading skeleton -->
    <div v-if="loading" class="animate-pulse">
      <div class="flex items-center justify-between mb-4">
        <div class="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
        <div class="w-16 h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
      <div class="w-24 h-4 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
      <div class="w-32 h-8 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
      <div class="w-28 h-3 bg-gray-200 dark:bg-gray-700 rounded"></div>
    </div>
    
    <!-- KPI Content -->
    <div v-else>
      <!-- Header with icon and growth -->
      <div class="flex items-center justify-between mb-4">
        <div :class="`p-3 rounded-lg bg-gradient-to-br ${getColorClasses(color).split(' ').slice(0, 2).join(' ')} shadow-sm`">
          <Icon :name="icon" class="w-8 h-8 text-white" />
        </div>
        
        <div v-if="growth !== 0" :class="`flex items-center px-2 py-1 rounded-full text-xs font-bold ${
          growth > 0 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' : 
          growth < 0 ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' :
          'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-300'
        }`">
          <Icon :name="growth > 0 ? 'heroicons:arrow-trending-up' : growth < 0 ? 'heroicons:arrow-trending-down' : 'heroicons:minus'" class="w-3 h-3 mr-1" />
          {{ formatGrowth(growth) }}
        </div>
      </div>
      
      <!-- Title -->
      <h3 class="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2 group-hover:text-gray-700 dark:group-hover:text-gray-200 transition-colors">
        {{ title }}
      </h3>
      
      <!-- Value -->
      <div class="text-3xl font-bold text-gray-900 dark:text-white mb-1 tracking-tight">
        {{ value }}
      </div>
      
      <!-- Subtitle -->
      <p v-if="subtitle" class="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
        {{ subtitle }}
      </p>
    </div>
  </div>
</template>