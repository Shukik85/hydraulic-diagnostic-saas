<!--
  pages/equipment/[id]/diagnostics.vue — Страница диагностики оборудования
  
  Интегрирует DiagnosticsDashboard с данными оборудования
-->
<template>
  <div class="diagnostics-page">
    <!-- Breadcrumbs -->
    <div class="mb-6">
      <nav class="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <NuxtLink to="/equipment" class="hover:text-gray-900 dark:hover:text-gray-100">
          Equipment
        </NuxtLink>
        <span>/</span>
        <NuxtLink
          :to="`/equipment/${route.params.id}`"
          class="hover:text-gray-900 dark:hover:text-gray-100"
        >
          {{ equipment?.name || 'Loading...' }}
        </NuxtLink>
        <span>/</span>
        <span class="text-gray-900 dark:text-gray-100">Diagnostics</span>
      </nav>
    </div>
    
    <!-- Loading state -->
    <div v-if="isLoading" class="flex items-center justify-center py-20">
      <div class="text-center">
        <UIcon name="i-heroicons-arrow-path" class="w-12 h-12 animate-spin text-blue-500 mb-4" />
        <p class="text-gray-500 dark:text-gray-400">Loading diagnostics data...</p>
      </div>
    </div>
    
    <!-- Error state -->
    <div v-else-if="error" class="py-20">
      <UAlert
        color="red"
        icon="i-heroicons-exclamation-triangle"
        title="Failed to load diagnostics"
        :description="error"
      >
        <template #actions>
          <UButton @click="loadData">Retry</UButton>
        </template>
      </UAlert>
    </div>
    
    <!-- Dashboard -->
    <DiagnosticsDashboard
      v-else-if="equipment && sensors.length > 0"
      :equipment-id="equipment.id"
      :components="components"
      :adjacency-matrix="adjacencyMatrix"
      :sensors="sensors"
    />
    
    <!-- No data state -->
    <div v-else class="py-20">
      <div class="text-center">
        <UIcon name="i-heroicons-chart-bar" class="w-16 h-16 text-gray-400 mb-4 mx-auto" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          No diagnostic data available
        </h3>
        <p class="text-gray-500 dark:text-gray-400 mb-6">
          Configure sensors and data sources to enable diagnostics
        </p>
        <UButton
          :to="`/equipment/${route.params.id}`"
          icon="i-heroicons-cog-6-tooth"
        >
          Configure Equipment
        </UButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import type { ComponentMetadata } from '~/types/metadata'

const route = useRoute()
const api = useApiAdvanced()
const metadataStore = useMetadataStore()

const isLoading = ref(true)
const error = ref<string | null>(null)
const equipment = ref<any>(null)
const components = ref<ComponentMetadata[]>([])
const adjacencyMatrix = ref<number[][]>([])
const sensors = ref<Array<{
  id: string
  name: string
  type: string
  unit: string
  expectedRange?: { min: number; max: number }
}>>([])

// Load equipment and diagnostic data
async function loadData() {
  isLoading.value = true
  error.value = null
  
  try {
    const equipmentId = route.params.id as string
    
    // Load equipment details
    const equipmentResponse = await api.get<any>(
      `/api/equipment/${equipmentId}`
    )
    equipment.value = equipmentResponse
    
    // Load metadata
    const metadataResponse = await api.get<any>(
      `/api/metadata/systems?equipment_id=${equipmentId}`
    )
    
    if (metadataResponse.system) {
      components.value = metadataResponse.system.components || []
      adjacencyMatrix.value = metadataResponse.system.adjacency_matrix || []
    }
    
    // Load sensor mappings
    const sensorsResponse = await api.get<any>(
      `/api/sensor-mappings?equipment_id=${equipmentId}`
    )
    
    // Transform sensor mappings to sensor list
    sensors.value = (sensorsResponse.mappings || []).map((mapping: any) => ({
      id: mapping.id,
      name: mapping.sensor_name || mapping.sensor_type,
      type: mapping.sensor_type,
      unit: mapping.unit || 'bar',
      expectedRange: mapping.expected_range ? {
        min: mapping.expected_range.min,
        max: mapping.expected_range.max
      } : undefined
    }))
    
  } catch (err: any) {
    console.error('Failed to load diagnostics data:', err)
    error.value = err.message || 'An error occurred while loading data'
  } finally {
    isLoading.value = false
  }
}

// Page metadata
definePageMeta({
  layout: 'default'
})

// SEO
useHead({
  title: computed(() => `Diagnostics - ${equipment.value?.name || 'Equipment'}`),
  meta: [
    {
      name: 'description',
      content: 'Real-time diagnostics and anomaly detection for hydraulic equipment'
    }
  ]
})

// Lifecycle
onMounted(() => {
  loadData()
})
</script>

<style scoped>
.diagnostics-page {
  @apply container mx-auto px-4 py-6;
}
</style>
