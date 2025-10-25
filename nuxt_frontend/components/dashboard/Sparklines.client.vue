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
  const padding = (max - min) * 0.15 || 1
  
  return {
    animation: true,
    animationDuration: 700,
    animationEasing: 'cubicOut',
    grid: {
      left: 0,
      right: 0,
      top: 4,
      bottom: 4
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
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const value = params[0].value
        return `${name}: ${value}${name === 'Температура' ? '°C' : name === 'Давление' ? ' бар' : name === 'Расход' ? ' л/мин' : ' мм/с'}`
      },
      axisPointer: {
        type: 'line',
        lineStyle: {
          color: color,
          width: 1
        }
      }
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
            { offset: 0, color: color + '60' },
            { offset: 1, color: color + '15' }
          ]
        }
      }
    }]
  }
}

const tempOption = computed(() => createSparklineOption(props.temp, '#ef4444', 'Температура'))
const pressureOption = computed(() => createSparklineOption(props.pressure, '#3b82f6', 'Давление'))
const flowOption = computed(() => createSparklineOption(props.flow, '#10b981', 'Расход'))
const vibrationOption = computed(() => createSparklineOption(props.vibration, '#a855f7', 'Вибрация'))
</script>

<template>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">🌡️ Температура</span>
        <span class="text-xs font-bold text-red-600 dark:text-red-400">{{ temp[temp.length-1] }}°C</span>
      </div>
      <div class="h-14">
        <ClientOnly>
          <VChart :option="tempOption" autoresize class="w-full h-full" />
        </ClientOnly>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">💧 Давление</span>
        <span class="text-xs font-bold text-blue-600 dark:text-blue-400">{{ pressure[pressure.length-1] }} бар</span>
      </div>
      <div class="h-14">
        <ClientOnly>
          <VChart :option="pressureOption" autoresize class="w-full h-full" />
        </ClientOnly>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">🌊 Расход</span>
        <span class="text-xs font-bold text-green-600 dark:text-green-400">{{ flow[flow.length-1] }} л/мин</span>
      </div>
      <div class="h-14">
        <ClientOnly>
          <VChart :option="flowOption" autoresize class="w-full h-full" />
        </ClientOnly>
      </div>
    </div>

    <div class="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500 dark:text-gray-400">⚡ Вибрация</span>
        <span class="text-xs font-bold text-purple-600 dark:text-purple-400">{{ vibration[vibration.length-1] }} мм/с</span>
      </div>
      <div class="h-14">
        <ClientOnly>
          <VChart :option="vibrationOption" autoresize class="w-full h-full" />
        </ClientOnly>
      </div>
    </div>
  </div>
</template>