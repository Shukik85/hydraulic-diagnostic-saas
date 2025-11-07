<template>
  <div class="flex items-center justify-between">
    <div class="flex-1">
      <slot />
    </div>
    <div class="w-20 h-8 ml-4">
      <VChart :option="sparklineOption" class="w-full h-full" autoresize />
    </div>
  </div>
</template>

<script setup lang="ts">
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { computed } from 'vue'

// Register minimal ECharts components for sparklines
use([LineChart, CanvasRenderer])

interface Props {
  data: number[]
  color?: string
  trend?: 'up' | 'down' | 'neutral'
}

const props = withDefaults(defineProps<Props>(), {
  color: '#3b82f6',
  trend: 'neutral'
})

// Minimal sparkline option
const sparklineOption = computed(() => ({
  grid: { left: 0, right: 0, top: 0, bottom: 0 },
  xAxis: {
    type: 'category',
    show: false,
    data: props.data.map((_, i) => i)
  },
  yAxis: {
    type: 'value',
    show: false
  },
  series: [{
    type: 'line',
    data: props.data,
    smooth: true,
    symbol: 'none',
    lineStyle: {
      color: props.color,
      width: 1.5
    },
    areaStyle: {
      color: {
        type: 'linear',
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: props.color + '30' },
          { offset: 1, color: props.color + '08' }
        ]
      }
    }
  }]
}))
</script>