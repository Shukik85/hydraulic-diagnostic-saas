<script setup lang='ts'>
const props = defineProps<{
  temp: number[]
  pressure: number[]
  flow: number[]
  vibration: number[]
}>()

// Simple CSS-based sparklines using SVG paths
const createSparklinePath = (data: number[]) => {
  if (!data || data.length === 0) return ''
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const width = 100
  const height = 30
  
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width
    const y = height - ((value - min) / range) * height
    return `${x},${y}`
  }).join(' ')
  
  return `M ${points.replace(/,/g, ' L ')}`
}

const tempPath = computed(() => createSparklinePath(props.temp))
const pressurePath = computed(() => createSparklinePath(props.pressure))
const flowPath = computed(() => createSparklinePath(props.flow))
const vibrationPath = computed(() => createSparklinePath(props.vibration))
</script>

<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Температура</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ temp[temp.length-1] }}°C</span>
      </div>
      <div class="h-8">
        <svg viewBox="0 0 100 30" class="w-full h-full">
          <defs>
            <linearGradient id="tempGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#ef4444;stop-opacity:0.3" />
              <stop offset="100%" style="stop-color:#ef4444;stop-opacity:0.1" />
            </linearGradient>
          </defs>
          <path :d="tempPath + ' L 100,30 L 0,30 Z'" fill="url(#tempGradient)" />
          <path :d="tempPath" stroke="#ef4444" stroke-width="2" fill="none" />
        </svg>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Давление</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ pressure[pressure.length-1] }} бар</span>
      </div>
      <div class="h-8">
        <svg viewBox="0 0 100 30" class="w-full h-full">
          <defs>
            <linearGradient id="pressureGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:0.3" />
              <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:0.1" />
            </linearGradient>
          </defs>
          <path :d="pressurePath + ' L 100,30 L 0,30 Z'" fill="url(#pressureGradient)" />
          <path :d="pressurePath" stroke="#3b82f6" stroke-width="2" fill="none" />
        </svg>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Расход</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ flow[flow.length-1] }} л/мин</span>
      </div>
      <div class="h-8">
        <svg viewBox="0 0 100 30" class="w-full h-full">
          <defs>
            <linearGradient id="flowGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#10b981;stop-opacity:0.3" />
              <stop offset="100%" style="stop-color:#10b981;stop-opacity:0.1" />
            </linearGradient>
          </defs>
          <path :d="flowPath + ' L 100,30 L 0,30 Z'" fill="url(#flowGradient)" />
          <path :d="flowPath" stroke="#10b981" stroke-width="2" fill="none" />
        </svg>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Вибрация</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ vibration[vibration.length-1] }} мм/с</span>
      </div>
      <div class="h-8">
        <svg viewBox="0 0 100 30" class="w-full h-full">
          <defs>
            <linearGradient id="vibrationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#a855f7;stop-opacity:0.3" />
              <stop offset="100%" style="stop-color:#a855f7;stop-opacity:0.1" />
            </linearGradient>
          </defs>
          <path :d="vibrationPath + ' L 100,30 L 0,30 Z'" fill="url(#vibrationGradient)" />
          <path :d="vibrationPath" stroke="#a855f7" stroke-width="2" fill="none" />
        </svg>
      </div>
    </div>
  </div>
</template>