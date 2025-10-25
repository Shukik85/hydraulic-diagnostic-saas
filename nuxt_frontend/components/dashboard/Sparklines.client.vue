<script setup lang='ts'>
const props = defineProps<{
  temp: number[]
  pressure: number[]
  flow: number[]
  vibration: number[]
}>()

// Create ECharts options for each sparkline
const createSparklineOption = (data: number[], color: string, name: string) => {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const padding = (max - min) * 0.1 || 1
  
  return {
    animation: true,
    animationDuration: 700,
    grid: {
      left: 0,
      right: 0,
      top: 2,
      bottom: 2
    },
    xAxis: {
      type: 'category',
      show: false,
      data: data.map((_, i) => i)
    },
    yAxis: {
      type: 'value',
      show: false,
      min: min - padding,
      max: max + padding
    },
    series: [{
      type: 'line',
      data,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 2.5,
        color
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: color + '40' },
            { offset: 1, color: color + '10' }
          ]
        }
      }
    }]
  }
}

const tempOption = computed(() => createSparklineOption(props.temp, '#ef4444', 'Temperature'))
const pressureOption = computed(() => createSparklineOption(props.pressure, '#3b82f6', 'Pressure'))
const flowOption = computed(() => createSparklineOption(props.flow, '#10b981', 'Flow'))
const vibrationOption = computed(() => createSparklineOption(props.vibration, '#a855f7', 'Vibration'))
</script>

<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Температура</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ temp[temp.length-1] }}°C</span>
      </div>
      <div class="h-14">
        <VChart :option="tempOption" autoresize />
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Давление</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ pressure[pressure.length-1] }} бар</span>
      </div>
      <div class="h-14">
        <VChart :option="pressureOption" autoresize />
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Расход</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ flow[flow.length-1] }} л/мин</span>
      </div>
      <div class="h-14">
        <VChart :option="flowOption" autoresize />
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">Вибрация</span>
        <span class="text-xs font-medium text-gray-900 dark:text-white">{{ vibration[vibration.length-1] }} мм/с</span>
      </div>
      <div class="h-14">
        <VChart :option="vibrationOption" autoresize />
      </div>
    </div>
  </div>
</template>