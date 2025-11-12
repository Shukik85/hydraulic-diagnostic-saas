<!--
  GraphView.vue — Force-directed graph визуализация архитектуры системы
  
  Features:
  - Interactive force-directed graph
  - Nodes = components
  - Edges = connections from adjacency matrix
  - Node color = anomaly score
  - Hover tooltips
  - Zoom & pan
  - Click to select component
  
  MIGRATED: BaseCard → UCard, StatusBadge → UBadge
-->
<template>
  <UCard class="graph-view p-6">
    <div class="flex items-center justify-between mb-4">
      <div>
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          System Architecture
        </h3>
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Interactive component graph
        </p>
      </div>
      
      <div class="flex items-center gap-2">
        <!-- Legend -->
        <div class="flex items-center gap-3 text-xs text-gray-700 dark:text-gray-300">
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded-full bg-green-500" />
            <span>Normal</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded-full bg-yellow-500" />
            <span>Warning</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded-full bg-red-500" />
            <span>Critical</span>
          </div>
        </div>
        
        <!-- Layout button -->
        <UButton
          icon="i-heroicons-arrow-path"
          size="sm"
          color="gray"
          variant="outline"
          @click="resetLayout"
        >
          Reset Layout
        </UButton>
      </div>
    </div>
    
    <!-- Graph -->
    <div class="graph-container" :style="{ height: graphHeight }">
      <ClientOnly>
        <v-chart
          ref="chartRef"
          :option="chartOption"
          autoresize
          class="w-full h-full"
          @click="onNodeClick"
        />
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
    
    <!-- Selected component info -->
    <div
      v-if="selectedComponent"
      class="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
    >
      <div class="flex items-center justify-between mb-3">
        <h4 class="font-semibold text-gray-900 dark:text-gray-100">
          {{ selectedComponent.name }}
        </h4>
        <UBadge
          :color="getComponentStatusColor(selectedComponent)"
          variant="soft"
        >
          {{ getComponentStatusLabel(selectedComponent) }}
        </UBadge>
      </div>
      
      <div class="grid grid-cols-3 gap-4 text-sm">
        <div>
          <p class="text-gray-500 dark:text-gray-400 mb-1">Type</p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ selectedComponent.component_type }}
          </p>
        </div>
        <div>
          <p class="text-gray-500 dark:text-gray-400 mb-1">Anomaly Score</p>
          <p class="font-medium" :class="getAnomalyScoreColor(selectedComponent.anomaly_score)">
            {{ selectedComponent.anomaly_score?.toFixed(2) || 'N/A' }}
          </p>
        </div>
        <div>
          <p class="text-gray-500 dark:text-gray-400 mb-1">Connections</p>
          <p class="font-medium text-gray-900 dark:text-gray-100">
            {{ selectedComponent.connected_to?.length || 0 }}
          </p>
        </div>
      </div>
      
      <div class="flex gap-2 mt-4">
        <UButton
          size="sm"
          color="primary"
          icon="i-heroicons-chart-bar"
          @click="viewComponentDetails"
        >
          View Details
        </UButton>
        <UButton
          size="sm"
          color="gray"
          variant="outline"
          @click="selectedComponent = null"
        >
          Close
        </UButton>
      </div>
    </div>
  </UCard>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import type { EChartsOption } from 'echarts'
import type { ComponentMetadata } from '~/types/metadata'

interface Props {
  components: ComponentMetadata[]
  adjacencyMatrix: number[][]
  anomalyScores?: Record<string, number>
  graphHeight?: string
}

const props = withDefaults(defineProps<Props>(), {
  graphHeight: '500px',
  anomalyScores: () => ({})
})

const emit = defineEmits<{
  componentSelect: [component: ComponentMetadata]
}>()

const chartRef = ref()
const selectedComponent = ref<ComponentMetadata | null>(null)

// Transform components to graph nodes
const graphNodes = computed(() => {
  return props.components.map((comp, index) => {
    const anomalyScore = props.anomalyScores[comp.id] || 0
    
    return {
      id: comp.id,
      name: comp.name || `Component ${index + 1}`,
      category: getComponentCategory(comp.component_type),
      value: anomalyScore,
      symbolSize: 40 + anomalyScore * 20, // Size based on anomaly
      itemStyle: {
        color: getNodeColor(anomalyScore)
      },
      label: {
        show: true,
        fontSize: 12,
        color: '#374151'
      },
      // Store full component data
      component: comp,
      anomaly_score: anomalyScore
    }
  })
})

// Transform adjacency matrix to graph edges
const graphEdges = computed(() => {
  const edges: any[] = []
  
  props.adjacencyMatrix.forEach((row, i) => {
    row.forEach((connected, j) => {
      if (connected && i !== j) {
        const sourceComp = props.components[i]
        const targetComp = props.components[j]
        
        if (sourceComp && targetComp) {
          edges.push({
            source: sourceComp.id,
            target: targetComp.id,
            lineStyle: {
              width: 2,
              curveness: 0.2,
              color: '#cbd5e1'
            }
          })
        }
      }
    })
  })
  
  return edges
})

// Component categories for legend
const categories = [
  { name: 'Pump' },
  { name: 'Valve' },
  { name: 'Cylinder' },
  { name: 'Filter' },
  { name: 'Tank' },
  { name: 'Other' }
]

// Chart option
const chartOption = computed<EChartsOption>(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    formatter: (params: any) => {
      if (params.dataType === 'node') {
        const comp = params.data.component
        const score = params.data.value
        return `
          <strong>${params.data.name}</strong><br/>
          Type: ${comp.component_type}<br/>
          Anomaly Score: ${score.toFixed(2)}<br/>
          Connections: ${comp.connected_to?.length || 0}
        `
      }
      return params.data.source + ' → ' + params.data.target
    }
  },
  legend: {
    data: categories.map(c => c.name),
    orient: 'vertical',
    right: 10,
    top: 'center',
    textStyle: {
      color: '#6b7280'
    }
  },
  series: [
    {
      type: 'graph',
      layout: 'force',
      data: graphNodes.value,
      links: graphEdges.value,
      categories: categories,
      roam: true,
      draggable: true,
      label: {
        position: 'bottom',
        show: true
      },
      force: {
        repulsion: 1000,
        edgeLength: 150,
        gravity: 0.1,
        friction: 0.6
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: {
          width: 4
        },
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.3)'
        }
      },
      lineStyle: {
        color: '#cbd5e1',
        width: 2,
        curveness: 0.2
      }
    }
  ]
}))

// Get node color based on anomaly score
function getNodeColor(score: number): string {
  if (score < 0.3) return '#22c55e' // green
  if (score < 0.7) return '#eab308' // yellow
  return '#ef4444' // red
}

// Get component category
function getComponentCategory(type: string): number {
  const categoryMap: Record<string, number> = {
    'pump': 0,
    'valve': 1,
    'cylinder': 2,
    'filter': 3,
    'tank': 4
  }
  return categoryMap[type.toLowerCase()] ?? 5
}

// Get component status
function getComponentStatus(comp: ComponentMetadata & { anomaly_score?: number }): string {
  const score = comp.anomaly_score || props.anomalyScores[comp.id] || 0
  
  if (score < 0.3) return 'operational'
  if (score < 0.7) return 'warning'
  return 'critical'
}

// Get component status color for UBadge
function getComponentStatusColor(comp: ComponentMetadata & { anomaly_score?: number }): string {
  const status = getComponentStatus(comp)
  const colorMap: Record<string, string> = {
    operational: 'green',
    warning: 'yellow',
    critical: 'red'
  }
  return colorMap[status] || 'gray'
}

// Get component status label
function getComponentStatusLabel(comp: ComponentMetadata & { anomaly_score?: number }): string {
  const status = getComponentStatus(comp)
  const labelMap: Record<string, string> = {
    operational: 'Normal',
    warning: 'Warning',
    critical: 'Critical'
  }
  return labelMap[status] || 'Unknown'
}

// Get anomaly score color
function getAnomalyScoreColor(score?: number): string {
  if (!score) return 'text-gray-500 dark:text-gray-400'
  if (score < 0.3) return 'text-green-600 dark:text-green-400'
  if (score < 0.7) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

// Handle node click
function onNodeClick(params: any) {
  if (params.dataType === 'node') {
    selectedComponent.value = params.data.component
    selectedComponent.value.anomaly_score = params.data.value
    emit('componentSelect', params.data.component)
  }
}

// Reset layout
function resetLayout() {
  if (chartRef.value) {
    const chart = chartRef.value
    // Force chart to recalculate layout
    chart.setOption(chartOption.value, true)
  }
}

// View component details
function viewComponentDetails() {
  if (selectedComponent.value) {
    // Navigate or emit event
    console.log('View details for:', selectedComponent.value)
  }
}

// Watch for component changes
watch(() => props.components, () => {
  selectedComponent.value = null
})
</script>

<style scoped>
.graph-container {
  @apply w-full;
}
</style>
