<template>
  <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
    <!-- Performance Chart -->
    <div class="u-card p-6">
      <h3 class="u-h5 mb-4">System Performance</h3>
      <VChart :option="performanceOption" class="h-64 w-full" autoresize />
    </div>
    
    <!-- Temperature Trends -->
    <div class="u-card p-6">
      <h3 class="u-h5 mb-4">Temperature Trends</h3>
      <VChart :option="temperatureOption" class="h-64 w-full" autoresize />
    </div>
    
    <!-- Pressure Analysis -->
    <div class="u-card p-6">
      <h3 class="u-h5 mb-4">Pressure Analysis</h3>
      <VChart :option="pressureOption" class="h-64 w-full" autoresize />
    </div>
  </div>
</template>

<script setup lang="ts">
import { use } from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { 
  GridComponent, 
  TooltipComponent, 
  LegendComponent 
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

// Register ECharts components
use([
  LineChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer
])

// Mock data
const performanceData = [
  { time: '00:00', value: 85 },
  { time: '04:00', value: 88 },
  { time: '08:00', value: 92 },
  { time: '12:00', value: 89 },
  { time: '16:00', value: 94 },
  { time: '20:00', value: 87 }
]

const temperatureData = [
  { time: '00:00', value: 65 },
  { time: '04:00', value: 62 },
  { time: '08:00', value: 68 },
  { time: '12:00', value: 72 },
  { time: '16:00', value: 75 },
  { time: '20:00', value: 69 }
]

const pressureData = [
  { time: '00:00', value: 2.1 },
  { time: '04:00', value: 2.3 },
  { time: '08:00', value: 2.2 },
  { time: '12:00', value: 2.4 },
  { time: '16:00', value: 2.2 },
  { time: '20:00', value: 2.1 }
]

// Chart options
const performanceOption = computed(() => ({
  grid: { left: 40, right: 20, top: 20, bottom: 40 },
  tooltip: { trigger: 'axis' },
  xAxis: {
    type: 'category',
    data: performanceData.map(d => d.time),
    axisLabel: { color: '#6b7280' }
  },
  yAxis: {
    type: 'value',
    axisLabel: { color: '#6b7280' },
    splitLine: { lineStyle: { color: '#f3f4f6' } }
  },
  series: [{
    type: 'line',
    data: performanceData.map(d => d.value),
    smooth: true,
    lineStyle: { color: '#10b981', width: 3 },
    areaStyle: {
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: '#10b98140' },
          { offset: 1, color: '#10b98110' }
        ]
      }
    }
  }]
}))

const temperatureOption = computed(() => ({
  grid: { left: 40, right: 20, top: 20, bottom: 40 },
  tooltip: { trigger: 'axis' },
  xAxis: {
    type: 'category',
    data: temperatureData.map(d => d.time),
    axisLabel: { color: '#6b7280' }
  },
  yAxis: {
    type: 'value',
    axisLabel: { color: '#6b7280' },
    splitLine: { lineStyle: { color: '#f3f4f6' } }
  },
  series: [{
    type: 'line',
    data: temperatureData.map(d => d.value),
    smooth: true,
    lineStyle: { color: '#f59e0b', width: 2 },
    symbol: 'circle',
    symbolSize: 6
  }]
}))

const pressureOption = computed(() => ({
  grid: { left: 40, right: 20, top: 20, bottom: 40 },
  tooltip: { trigger: 'axis' },
  xAxis: {
    type: 'category',
    data: pressureData.map(d => d.time),
    axisLabel: { color: '#6b7280' }
  },
  yAxis: {
    type: 'value',
    axisLabel: { color: '#6b7280' },
    splitLine: { lineStyle: { color: '#f3f4f6' } }
  },
  series: [{
    type: 'bar',
    data: pressureData.map(d => d.value),
    itemStyle: {
      color: '#8b5cf6',
      borderRadius: [4, 4, 0, 0]
    }
  }]
}))
</script>