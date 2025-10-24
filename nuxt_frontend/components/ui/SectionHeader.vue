<script setup lang="ts">
interface Props {
  title: string
  description?: string
  icon?: string
  iconColor?: 'blue' | 'green' | 'purple' | 'orange' | 'teal' | 'red' | 'indigo'
  actionText?: string
  actionHref?: string
  badge?: string
  badgeColor?: 'blue' | 'green' | 'purple' | 'orange' | 'teal' | 'red' | 'indigo'
}

withDefaults(defineProps<Props>(), {
  iconColor: 'blue',
  badgeColor: 'blue'
})

const getIconColorClass = (color: string) => {
  const colorMap = {
    blue: 'text-blue-600 dark:text-blue-400',
    green: 'text-green-600 dark:text-green-400',
    purple: 'text-purple-600 dark:text-purple-400',
    orange: 'text-orange-600 dark:text-orange-400',
    teal: 'text-teal-600 dark:text-teal-400',
    red: 'text-red-600 dark:text-red-400',
    indigo: 'text-indigo-600 dark:text-indigo-400'
  }
  return colorMap[color] || colorMap.blue
}

const getBadgeColorClass = (color: string) => {
  const colorMap = {
    blue: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
    green: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300',
    purple: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300',
    orange: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300',
    teal: 'bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300',
    red: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
    indigo: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
  }
  return colorMap[color] || colorMap.blue
}
</script>

<template>
  <div class="flex items-center justify-between mb-6">
    <div class="flex-1">
      <div class="flex items-center space-x-4 mb-2">
        <!-- Icon -->
        <Icon 
          v-if="icon" 
          :name="icon" 
          :class="`w-6 h-6 ${getIconColorClass(iconColor)}`" 
        />
        
        <!-- Title -->
        <h2 class="text-xl font-bold text-gray-900 dark:text-white">
          {{ title }}
        </h2>
        
        <!-- Badge -->
        <span 
          v-if="badge"
          :class="`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getBadgeColorClass(badgeColor)}`"
        >
          {{ badge }}
        </span>
      </div>
      
      <!-- Description -->
      <p v-if="description" class="text-gray-600 dark:text-gray-300">
        {{ description }}
      </p>
    </div>
    
    <!-- Action button -->
    <div v-if="actionText && actionHref">
      <NuxtLink
        :to="actionHref"
        class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-200 hover:scale-105"
      >
        {{ actionText }}
        <Icon name="heroicons:arrow-right" class="w-4 h-4 ml-2" />
      </NuxtLink>
    </div>
  </div>
</template>