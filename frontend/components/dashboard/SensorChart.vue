<template>
  <div class="u-chart-wrapper">
    <div class="u-chart-header">
      <h3 class="u-chart-title">{{ title }}</h3>
      <div class="u-chart-controls">
        <select 
          v-model="timeRange"
          class="u-input text-sm py-1 px-2 w-32"
          @change="updateChart"
        >
          <option value="1h">Last Hour</option>
          <option value="24h">Last 24h</option>
          <option value="7d">Last 7 days</option>
        </select>
      </div>
    </div>
    
    <div class="u-chart-container">
      <ClientOnly>
        <component 
          :is="VChart"
          v-if="chartOptions"
          :option="chartOptions" 
          :autoresize="true"
          class="w-full h-full"
        />
        <template #fallback>
          <div class="u-flex-center h-full">
            <div class="u-spinner w-8 h-8"></div>
            <span class="ml-2 u-body text-gray-500 dark:text-gray-400">Loading chart...</span>
          </div>
        </template>
      </ClientOnly>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title: string
  type: 'line' | 'bar' | 'gauge' | 'pie'
  data: any[]
  unit?: string
}

const props = withDefaults(defineProps<Props>(), {
  unit: ''
})

const { $VChart: VChart } = useNuxtApp()
const timeRange = ref('24h')

const chartOptions = computed(() => {
  if (!props.data?.length) return null
  
  const baseOptions = {
    backgroundColor: 'transparent',
    textStyle: {
      fontFamily: 'var(--font-sans)',
      fontSize: 12
    }
  }
  
  switch (props.type) {
    case 'line':
      return {
        ...baseOptions,
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderColor: '#e5e7eb',
          borderRadius: 8,
          textStyle: { color: '#374151', fontSize: 13 }
        },
        grid: {
          left: '3%',
          right: '4%', 
          bottom: '15%',
          top: '15%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: props.data.map(item => item.time),
          axisLine: { lineStyle: { color: '#e5e7eb' } },
          axisTick: { show: false },
          axisLabel: { color: '#6b7280', fontSize: 11 }
        },
        yAxis: {
          type: 'value',
          name: props.unit,
          axisLine: { show: false },
          splitLine: { lineStyle: { color: '#f3f4f6', type: 'dashed' } },
          axisLabel: { color: '#6b7280', fontSize: 11 }
        },
        series: [{
          type: 'line',
          data: props.data.map(item => item.value),
          smooth: true,
          lineStyle: { color: '#2563eb', width: 3 },
          itemStyle: { 
            color: '#2563eb', 
            borderWidth: 2,
            borderColor: '#ffffff'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(37, 99, 235, 0.15)' },
                { offset: 1, color: 'rgba(37, 99, 235, 0.02)' }
              ]
            }
          }
        }]
      }
      
    case 'gauge':
      return {
        ...baseOptions,
        series: [{
          type: 'gauge',
          center: ['50%', '60%'],
          startAngle: 200,
          endAngle: -40,
          min: 0,
          max: props.data[0]?.max || 100,
          splitNumber: 10,
          itemStyle: {
            color: '#2563eb'
          },
          progress: {
            show: true,
            width: 25
          },
          pointer: {
            show: false
          },
          axisLine: {
            lineStyle: {
              width: 25,
              color: [[1, '#e5e7eb']]
            }
          },
          axisTick: {
            distance: -40,
            splitNumber: 2,
            lineStyle: {
              width: 2,
              color: '#9ca3af'
            }
          },
          splitLine: {
            distance: -45,
            length: 12,
            lineStyle: {
              width: 2,
              color: '#9ca3af'
            }
          },
          axisLabel: {
            distance: -20,
            color: '#6b7280',
            fontSize: 11
          },
          anchor: {
            show: false
          },
          title: {
            show: false
          },
          detail: {
            valueAnimation: true,
            width: '60%',
            lineHeight: 40,
            borderRadius: 8,
            offsetCenter: [0, '-10%'],
            fontSize: 22,
            fontWeight: 'bold',
            formatter: `{value} ${props.unit}`,
            color: '#2563eb'
          },
          data: [{
            value: props.data[0]?.value || 0
          }]
        }]
      }
      
    default:
      return {
        ...baseOptions,
        tooltip: { trigger: 'item' },
        series: [{
          type: props.type,
          data: props.data
        }]
      }
  }
})

const updateChart = () => {
  console.log('Update chart for timeRange:', timeRange.value)
}
</script>