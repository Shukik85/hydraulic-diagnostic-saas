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
  grid: {
    left: 40,
    right: 20,
    top: 20,
    bottom: 40
  },
  xAxis: {
    type: 'category',
    data: trendData.value.map(d => d.name),
    axisLine: { lineStyle: { color: '#e5e7eb' } },
    axisLabel: { color: '#9ca3af' }
  },
  yAxis: {
    type: 'value',
    min: 99.6,
    max: 100,
    axisLine: { lineStyle: { color: '#e5e7eb' } },
    axisLabel: { color: '#9ca3af' },
    splitLine: { lineStyle: { color: '#f3f4f6' } }
  },
  tooltip: {
    trigger: 'axis',
    formatter: '{b}: {c}%'
  },
  series: [{
    type: 'line',
    data: trendData.value.map(d => d.uptime),
    smooth: true,
    symbol: 'circle',
    symbolSize: 6,
    lineStyle: {
      width: 3,
      color: '#3b82f6'
    },
    areaStyle: {
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: '#3b82f680' },
          { offset: 1, color: '#3b82f620' }
        ]
      }
    }
  }]
}))

const alertsOption = computed(() => ({
  animation: true,
  animationDuration: 600,
  grid: {
    left: 40,
    right: 20,
    top: 20,
    bottom: 40
  },
  xAxis: {
    type: 'category',
    data: trendData.value.map(d => d.name),
    axisLine: { lineStyle: { color: '#e5e7eb' } },
    axisLabel: { color: '#9ca3af' }
  },
  yAxis: {
    type: 'value',
    min: 0,
    axisLine: { lineStyle: { color: '#e5e7eb' } },
    axisLabel: { color: '#9ca3af' },
    splitLine: { lineStyle: { color: '#f3f4f6' } }
  },
  tooltip: {
    trigger: 'axis',
    formatter: '{b}: {c} алертов'
  },
  series: [{
    type: 'bar',
    data: trendData.value.map(d => d.alerts),
    itemStyle: {
      color: '#ef4444',
      borderRadius: [8, 8, 0, 0]
    },
    barWidth: '60%'
  }]
}))
</script>

<template>
  <div class="premium-card p-6 mt-6">
    <h3 class="premium-heading-sm text-gray-900 dark:text-white mb-4">📊 Тренды за неделю</h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Uptime chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
        <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Время работы (%)</div>
        <div class="h-60">
          <VChart :option="uptimeOption" autoresize />
        </div>
      </div>

      <!-- Alerts chart -->
      <div class="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
        <div class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Алерты (шт.)</div>
        <div class="h-60">
          <VChart :option="alertsOption" autoresize />
        </div>
      </div>
    </div>
  </div>
</template>