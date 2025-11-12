<!--
  SensorChart.vue — Time-series график для данных сенсоров
  
  Features:
  - Configurable refresh interval (10s/30s/60s/Manual)
  - Expected range zones
  - Anomaly markers
  - Zoom & pan
  - Export to image
  
  MIGRATED: BaseCard → UCard, BaseButton → UButton
-->
<template>
  <UCard class="sensor-chart p-6">
    <div class="flex items-center justify-between mb-4">
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ sensorName }}
        </h3>
        <p class="text-sm text-gray-500 dark:text-gray-400">
          {{ sensorType }} - {{ unit }}
        </p>
      </div>
      
      <div class="flex items-center gap-2">
        <!-- Refresh interval selector -->
        <USelectMenu
          v-model="refreshInterval"
          :options="refreshOptions"
          size="sm"
          @update:model-value="onRefreshIntervalChange"
        />
        
        <!-- Manual refresh button -->
        <UButton
          icon="i-heroicons-arrow-path"
          size="sm"
          color="gray"
          variant="outline"
          :loading="isLoading"
          @click="fetchData"
        >
          Refresh
        </UButton>
      </div>
    </div>
    
    <!-- Chart -->
    <div class="chart-container" :style="{ height: chartHeight }">
      <ClientOnly>
        <v-chart
          v-if="data.length > 0"
          :option="chartOption"
          :loading="isLoading"
          autoresize
          class="w-full h-full"
          @click="onChartClick"
        />
        <div v-else class="flex items-center justify-center h-full">
          <div class="text-center">
            <UIcon
              name="i-heroicons-chart-bar"
              class="w-16 h-16 text-gray-300 dark:text-gray-700 mx-auto mb-3"
            />
            <p class="text-sm text-gray-500 dark:text-gray-400">
              No data available
            </p>
          </div>
        </div>
        <template #fallback>
          <div class="flex items-center justify-center h-full">
            <UIcon
              name="i-heroicons-arrow-path"
              class="w-8 h-8 animate-spin text-blue-500"
            />
          </div>
        </template>
      </ClientOnly>
    </div>
    
    <!-- Stats -->
    <div class="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Current</p>
        <p class="text-lg font-semibold" :class="getCurrentValueColor()">
          {{ currentValue?.toFixed(2) || 'N/A' }}
        </p>
      </div>
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Min</p>
        <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ stats.min?.toFixed(2) || 'N/A' }}
        </p>
      </div>
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Max</p>
        <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ stats.max?.toFixed(2) || 'N/A' }}
        </p>
      </div>
      <div>
        <p class="text-xs text-gray-500 dark:text-gray-400">Avg</p>
        <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ stats.avg?.toFixed(2) || 'N/A' }}
        </p>
      </div>
    </div>
  </UCard>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import type { EChartsOption } from 'echarts'

interface Props {
  sensorId: string
  sensorName: string
  sensorType: string
  unit?: string
  expectedRange?: { min: number; max: number }
  chartHeight?: string
  timeRange?: number // minutes
}

const props = withDefaults(defineProps<Props>(), {
  unit: 'bar',
  chartHeight: '300px',
  timeRange: 60 // last 60 minutes
})

const emit = defineEmits<{
  anomalyClick: [data: any]
}>()

const api = useApiAdvanced()
const isLoading = ref(false)
const data = ref<Array<{ timestamp: number; value: number; isAnomaly?: boolean }>>([]))
const currentValue = ref<number | null>(null)

// Refresh interval options
const refreshOptions = [
  { label: '10s', value: 10000 },
  { label: '30s', value: 30000 },
  { label: '1min', value: 60000 },
  { label: 'Manual', value: 0 }
]

const refreshInterval = ref(30000) // default 30s
let refreshTimer: ReturnType<typeof setInterval> | null = null

// Stats
const stats = computed(() => {
  if (data.value.length === 0) {
    return { min: null, max: null, avg: null }
  }
  
  const values = data.value.map(d => d.value)
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    avg: values.reduce((a, b) => a + b, 0) / values.length
  }
})

// Chart option
const chartOption = computed<EChartsOption>(() => {
  const seriesData = data.value.map(d => [d.timestamp, d.value])
  const anomalyData = data.value
    .filter(d => d.isAnomaly)
    .map(d => [d.timestamp, d.value])
  
  return {
    backgroundColor: 'transparent',
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '10%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: (params: any) => {
        const param = Array.isArray(params) ? params[0] : params
        const date = new Date(param.value[0])
        return `${date.toLocaleTimeString()}<br/>${param.value[1].toFixed(2)} ${props.unit}`
      }
    },
    xAxis: {
      type: 'time',
      splitLine: {
        show: false
      }
    },
    yAxis: {
      type: 'value',
      name: props.unit,
      splitLine: {
        lineStyle: {
          type: 'dashed',
          color: '#e5e7eb'
        }
      }
    },
    series: [
      // Main line
      {
        name: props.sensorName,
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          color: '#3b82f6'
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.2)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0)' }
            ]
          }
        },
        data: seriesData,
        // Expected range zone
        markArea: props.expectedRange ? {
          silent: true,
          itemStyle: {
            color: 'rgba(34, 197, 94, 0.1)'
          },
          data: [[
            {
              yAxis: props.expectedRange.min
            },
            {
              yAxis: props.expectedRange.max
            }
          ]]
        } : undefined
      },
      // Anomaly points
      {
        name: 'Anomalies',
        type: 'scatter',
        symbolSize: 10,
        itemStyle: {
          color: '#ef4444'
        },
        data: anomalyData,
        z: 10
      }
    ],
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100
      },
      {
        start: 0,
        end: 100,
        height: 20
      }
    ],
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: 'none'
        },
        restore: {},
        saveAsImage: {
          name: `${props.sensorName}_${Date.now()}`
        }
      }
    }
  }
})

// Fetch sensor data
async function fetchData() {
  isLoading.value = true
  
  try {
    const endTime = Date.now()
    const startTime = endTime - props.timeRange * 60 * 1000
    
    const response = await api.get<any>(
      `/api/sensors/${props.sensorId}/readings?start=${startTime}&end=${endTime}`
    )
    
    data.value = response.readings || []
    currentValue.value = data.value[data.value.length - 1]?.value || null
  } catch (error) {
    console.error('Failed to fetch sensor data:', error)
  } finally {
    isLoading.value = false
  }
}

// Handle refresh interval change
function onRefreshIntervalChange(interval: number) {
  // Save to localStorage
  if (process.client) {
    localStorage.setItem(
      `sensor_refresh_${props.sensorId}`,
      interval.toString()
    )
  }
  
  // Clear existing timer
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
  
  // Set new timer if not manual
  if (interval > 0) {
    refreshTimer = setInterval(fetchData, interval)
  }
}

// Get current value color
function getCurrentValueColor(): string {
  if (!currentValue.value || !props.expectedRange) {
    return 'text-gray-900 dark:text-gray-100'
  }
  
  if (
    currentValue.value >= props.expectedRange.min &&
    currentValue.value <= props.expectedRange.max
  ) {
    return 'text-green-600 dark:text-green-400'
  }
  
  return 'text-red-600 dark:text-red-400'
}

// Chart click handler
function onChartClick(params: any) {
  if (params.seriesName === 'Anomalies') {
    emit('anomalyClick', params.data)
  }
}

// Lifecycle
onMounted(() => {
  // Load saved refresh interval
  if (process.client) {
    const saved = localStorage.getItem(`sensor_refresh_${props.sensorId}`)
    if (saved) {
      refreshInterval.value = parseInt(saved, 10)
    }
  }
  
  // Initial fetch
  fetchData()
  
  // Start auto-refresh
  if (refreshInterval.value > 0) {
    refreshTimer = setInterval(fetchData, refreshInterval.value)
  }
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>

<style scoped>
.chart-container {
  @apply w-full;
}
</style>
