<template>
  <div class="h-full w-full">
    <VChart 
      ref="chartRef"
      :option="option" 
      autoresize 
      :class="props.class || 'h-40 w-full'"
    />
  </div>
</template>

<script setup lang="ts">
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { 
  GridComponent, 
  TooltipComponent, 
  LegendComponent 
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { ref, computed } from 'vue'

// Register ECharts components
use([
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer
])

interface Props {
  data: Array<{ name: string; value: number; [key: string]: any }>
  dataKey?: string
  xKey?: string
  color?: string
  showGrid?: boolean
  showTooltip?: boolean
  height?: string
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  dataKey: 'value',
  xKey: 'name',
  color: '#3b82f6',
  showGrid: true,
  showTooltip: true,
  height: '320px'
})

const chartRef = ref()

// ECharts option
const option = computed(() => ({
  grid: {
    left: props.showGrid ? 40 : 8,
    right: props.showGrid ? 20 : 8,
    top: props.showGrid ? 20 : 8,
    bottom: props.showGrid ? 40 : 8,
    containLabel: true
  },
  tooltip: props.showTooltip ? {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    },
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e5e7eb',
    borderWidth: 1,
    textStyle: {
      color: '#374151'
    }
  } : undefined,
  xAxis: {
    type: 'category',
    data: props.data.map(item => item[props.xKey]),
    axisLine: {
      show: props.showGrid,
      lineStyle: { color: '#e5e7eb' }
    },
    axisTick: {
      show: props.showGrid,
      lineStyle: { color: '#e5e7eb' }
    },
    axisLabel: {
      show: props.showGrid,
      color: '#6b7280'
    }
  },
  yAxis: {
    type: 'value',
    axisLine: {
      show: props.showGrid,
      lineStyle: { color: '#e5e7eb' }
    },
    axisTick: {
      show: props.showGrid,
      lineStyle: { color: '#e5e7eb' }
    },
    axisLabel: {
      show: props.showGrid,
      color: '#6b7280'
    },
    splitLine: {
      show: props.showGrid,
      lineStyle: {
        color: '#f3f4f6',
        type: 'dashed'
      }
    }
  },
  series: [{
    type: 'bar',
    data: props.data.map(item => item[props.dataKey]),
    itemStyle: {
      color: props.color,
      borderRadius: [4, 4, 0, 0]
    },
    emphasis: {
      focus: 'series',
      itemStyle: {
        shadowBlur: 10,
        shadowColor: props.color + '50'
      }
    },
    animationDuration: 1000,
    animationEasing: 'cubicOut'
  }]
}))
</script>