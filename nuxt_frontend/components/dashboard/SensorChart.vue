<template>
  <div class="premium-card p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="premium-heading-md text-gray-900 dark:text-white">{{ title }}</h3>
      <div class="flex items-center space-x-2">
        <select 
          v-model="timeRange"
          class="premium-input !py-2"
          @change="updateChart"
        >
          <option value="1h">Last Hour</option>
          <option value="24h">Last 24h</option>
          <option value="7d">Last 7 days</option>
        </select>
      </div>
    </div>
    
    <div class="h-80">
      <ClientOnly>
        <component 
          :is="VChart"
          v-if="chartOptions"
          :option="chartOptions" 
          :autoresize="true"
          class="w-full h-full"
        />
        <template #fallback>
          <div class="flex items-center justify-center h-full">
            <div class="inline-block w-6 h-6 rounded-full border-2 border-solid border-current border-r-transparent text-blue-600" style="animation: spin 1s linear infinite;"></div>
            <span class="ml-2 premium-body text-gray-500 dark:text-gray-400">Loading chart...</span>
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
      fontFamily: 'var(--font-sans)'
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
          textStyle: { color: '#374151' }
        },
        grid: {
          left: '3%',
          right: '4%', 
          bottom: '10%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: props.data.map(item => item.time),
          axisLine: { lineStyle: { color: '#e5e7eb' } },
          axisTick: { lineStyle: { color: '#e5e7eb' } },
          axisLabel: { color: '#6b7280' }
        },
        yAxis: {
          type: 'value',
          name: props.unit,
          axisLine: { lineStyle: { color: '#e5e7eb' } },
          splitLine: { lineStyle: { color: '#f3f4f6' } },
          axisLabel: { color: '#6b7280' }
        },
        series: [{
          type: 'line',
          data: props.data.map(item => item.value),
          smooth: true,
          lineStyle: { color: 'var(--color-primary)', width: 2 },
          itemStyle: { color: 'var(--color-primary)' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'var(--color-primary-light)' },
                { offset: 1, color: 'rgba(37, 99, 235, 0.05)' }
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
          splitNumber: 12,
          itemStyle: {
            color: 'var(--color-primary)'
          },
          progress: {
            show: true,
            width: 30
          },
          pointer: {
            show: false
          },
          axisLine: {
            lineStyle: {
              width: 30
            }
          },
          axisTick: {
            distance: -45,
            splitNumber: 5,
            lineStyle: {
              width: 2,
              color: '#999'
            }
          },
          splitLine: {
            distance: -52,
            length: 14,
            lineStyle: {
              width: 3,
              color: '#999'
            }
          },
          axisLabel: {
            distance: -20,
            color: '#999',
            fontSize: 12
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
            offsetCenter: [0, '-15%'],
            fontSize: 20,
            fontWeight: 'bold',
            formatter: `{value} ${props.unit}`,
            color: 'inherit'
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

<style scoped>
@keyframes spin { to { transform: rotate(360deg); } }
</style>
