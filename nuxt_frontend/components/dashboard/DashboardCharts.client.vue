<script setup lang='ts'>
const trendData = ref([
  { name: 'Пн', uptime: 99.7, alerts: 2 },
  { name: 'Вт', uptime: 99.9, alerts: 1 },
  { name: 'Ср', uptime: 99.8, alerts: 3 },
  { name: 'Чт', uptime: 99.95, alerts: 1 },
  { name: 'Пт', uptime: 99.92, alerts: 2 },
  { name: 'Сб', uptime: 99.96, alerts: 1 },
  { name: 'Вс', uptime: 99.94, alerts: 1 }
])

const uptimeOption = computed(() => ({
  animation: true,
  animationDuration: 800,
  animationEasing: 'cubicOut',
  grid: {
    left: 50,
    right: 30,
    top: 30,
    bottom: 50
  },
  xAxis: {
    type: 'category',
    data: trendData.value.map(d => d.name),
    axisLine: { 
      lineStyle: { color: '#e5e7eb' } 
    },
    axisLabel: { 
      color: '#6b7280',
      fontSize: 12,
      fontWeight: 500
    },
    axisTick: {
      alignWithLabel: true,
      lineStyle: { color: '#e5e7eb' }
    }
  },
  yAxis: {
    type: 'value',
    min: 99.6,
    max: 100,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { 
      color: '#6b7280',
      fontSize: 12,
      formatter: '{value}%'
    },
    splitLine: { 
      lineStyle: { 
        color: '#f3f4f6',
        type: 'dashed'
      } 
    }
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderWidth: 0,
    textStyle: {
      color: '#fff',
      fontSize: 12
    },
    formatter: (params: any) => {
      const data = params[0]
      return `${data.name}: <strong>${data.value}%</strong> время работы`
    }
  },
  series: [{
    type: 'line',
    data: trendData.value.map(d => d.uptime),
    smooth: true,
    symbol: 'circle',
    symbolSize: 8,
    lineStyle: {
      width: 3,
      color: '#3b82f6'
    },
    itemStyle: {
      color: '#3b82f6',
      borderColor: '#ffffff',
      borderWidth: 2
    },
    areaStyle: {
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: 'rgba(59, 130, 246, 0.8)' },
          { offset: 1, color: 'rgba(59, 130, 246, 0.1)' }
        ]
      }
    }
  }]
}))

const alertsOption = computed(() => ({
  animation: true,
  animationDuration: 600,
  animationEasing: 'cubicOut',
  grid: {
    left: 50,
    right: 30,
    top: 30,
    bottom: 50
  },
  xAxis: {
    type: 'category',
    data: trendData.value.map(d => d.name),
    axisLine: { 
      lineStyle: { color: '#e5e7eb' } 
    },
    axisLabel: { 
      color: '#6b7280',
      fontSize: 12,
      fontWeight: 500
    },
    axisTick: {
      alignWithLabel: true,
      lineStyle: { color: '#e5e7eb' }
    }
  },
  yAxis: {
    type: 'value',
    min: 0,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { 
      color: '#6b7280',
      fontSize: 12
    },
    splitLine: { 
      lineStyle: { 
        color: '#f3f4f6',
        type: 'dashed'
      } 
    }
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderWidth: 0,
    textStyle: {
      color: '#fff',
      fontSize: 12
    },
    formatter: (params: any) => {
      const data = params[0]
      const plural = data.value === 1 ? 'алерт' : data.value < 5 ? 'алерта' : 'алертов'
      return `${data.name}: <strong>${data.value}</strong> ${plural}`
    }
  },
  series: [{
    type: 'bar',
    data: trendData.value.map(d => d.alerts),
    itemStyle: {
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: '#ef4444' },
          { offset: 1, color: '#dc2626' }
        ]
      },
      borderRadius: [6, 6, 0, 0]
    },
    barWidth: '50%',
    emphasis: {
      itemStyle: {
        color: '#b91c1c'
      }
    }
  }]
}))
</script>

<template>
  <div class="premium-card p-6 mt-6">
    <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-6 flex items-center">
      <Icon name="heroicons:chart-bar" class="w-6 h-6 text-blue-600 mr-3" />
      📊 Тренды за неделю
    </h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Uptime chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
        <div class="flex items-center justify-between mb-4">
          <div class="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center">
            <div class="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            Время работы (%)
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">За последние 7 дней</div>
        </div>
        <div class="h-64">
          <ClientOnly>
            <VChart :option="uptimeOption" autoresize class="w-full h-full" />
          </ClientOnly>
        </div>
      </div>

      <!-- Alerts chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
        <div class="flex items-center justify-between mb-4">
          <div class="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center">
            <div class="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
            Алерты (шт.)
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Критические события</div>
        </div>
        <div class="h-64">
          <ClientOnly>
            <VChart :option="alertsOption" autoresize class="w-full h-full" />
          </ClientOnly>
        </div>
      </div>
    </div>
  </div>
</template>