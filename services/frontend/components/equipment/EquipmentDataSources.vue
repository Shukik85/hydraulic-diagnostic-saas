<!--
  EquipmentDataSources.vue — Таб с источниками данных
  
  Features:
  - Список всех источников (CSV, IoT Gateway, API, Simulator)
  - Статус каждого источника
  - Последняя синхронизация
  - Quick actions
  - Add new source
-->
<template>
  <div class="equipment-data-sources space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Data Sources
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
          {{ dataSources.length }} sources configured
        </p>
      </div>
      
      <UButton
        color="primary"
        icon="i-heroicons-plus"
        @click="openAddSourceModal"
      >
        Add Source
      </UButton>
    </div>
    
    <!-- Loading state -->
    <div v-if="isLoading" class="space-y-4">
      <USkeleton class="h-32 w-full" />
      <USkeleton class="h-32 w-full" />
      <USkeleton class="h-32 w-full" />
    </div>
    
    <!-- Empty state -->
    <UCard v-else-if="dataSources.length === 0" class="p-12">
      <div class="text-center">
        <UIcon
          name="i-heroicons-cloud-arrow-down"
          class="w-16 h-16 text-gray-400 dark:text-gray-600 mx-auto mb-4"
        />
        <h4 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          No data sources configured
        </h4>
        <p class="text-sm text-gray-600 dark:text-gray-400 mb-6">
          Connect data sources to start ingesting sensor data
        </p>
        <UButton
          color="primary"
          icon="i-heroicons-plus"
          @click="openAddSourceModal"
        >
          Add First Source
        </UButton>
      </div>
    </UCard>
    
    <!-- Data sources grid -->
    <div v-else class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <UCard
        v-for="source in dataSources"
        :key="source.id"
        class="p-6"
      >
        <!-- Header -->
        <div class="flex items-start justify-between mb-4">
          <div class="flex items-center gap-3">
            <div
              class="w-12 h-12 rounded-lg flex items-center justify-center"
              :class="getSourceIconBg(source.type)"
            >
              <UIcon
                :name="getSourceIcon(source.type)"
                class="w-6 h-6"
                :class="getSourceIconColor(source.type)"
              />
            </div>
            <div>
              <h4 class="font-semibold text-gray-900 dark:text-gray-100">
                {{ source.name }}
              </h4>
              <p class="text-xs text-gray-500 dark:text-gray-400">
                {{ formatSourceType(source.type) }}
              </p>
            </div>
          </div>
          
          <UBadge
            :color="source.status === 'active' ? 'green' : 'gray'"
            variant="soft"
          >
            <div class="flex items-center gap-1">
              <div
                class="w-1.5 h-1.5 rounded-full"
                :class="source.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'"
              />
              {{ source.status === 'active' ? 'Active' : 'Inactive' }}
            </div>
          </UBadge>
        </div>
        
        <!-- Stats -->
        <div class="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p class="text-xs text-gray-500 dark:text-gray-400 mb-1">
              Records Ingested
            </p>
            <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {{ formatNumber(source.records_count || 0) }}
            </p>
          </div>
          <div>
            <p class="text-xs text-gray-500 dark:text-gray-400 mb-1">
              Last Sync
            </p>
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ formatTime(source.last_sync) }}
            </p>
          </div>
        </div>
        
        <!-- Connection info -->
        <div class="space-y-2 mb-4">
          <div class="flex items-center gap-2 text-xs">
            <UIcon name="i-heroicons-server" class="w-3 h-3 text-gray-400" />
            <span class="text-gray-600 dark:text-gray-400">
              {{ source.endpoint || source.file_path || source.config?.gateway_url || 'N/A' }}
            </span>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <UIcon name="i-heroicons-clock" class="w-3 h-3 text-gray-400" />
            <span class="text-gray-600 dark:text-gray-400">
              Polling: {{ source.polling_interval || 'Manual' }}
            </span>
          </div>
        </div>
        
        <!-- Actions -->
        <div class="flex gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
          <UButton
            size="sm"
            color="gray"
            variant="outline"
            icon="i-heroicons-arrow-path"
            :loading="syncingIds.includes(source.id)"
            @click="syncSource(source)"
          >
            Sync Now
          </UButton>
          <UButton
            size="sm"
            color="gray"
            variant="ghost"
            icon="i-heroicons-pencil"
            @click="editSource(source)"
          />
          <UButton
            size="sm"
            color="red"
            variant="ghost"
            icon="i-heroicons-trash"
            @click="deleteSource(source)"
          />
        </div>
      </UCard>
    </div>
    
    <!-- Add Source Modal -->
    <UModal v-model="isAddModalOpen" :ui="{ width: 'sm:max-w-2xl' }">
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">
              Add Data Source
            </h3>
            <UButton
              color="gray"
              variant="ghost"
              icon="i-heroicons-x-mark"
              @click="isAddModalOpen = false"
            />
          </div>
        </template>
        
        <div class="space-y-4">
          <UFormGroup label="Source Name" required>
            <UInput
              v-model="newSource.name"
              placeholder="e.g., Production CSV Import"
            />
          </UFormGroup>
          
          <UFormGroup label="Source Type" required>
            <USelect
              v-model="newSource.type"
              :options="sourceTypeOptions"
            />
          </UFormGroup>
          
          <!-- CSV specific -->
          <template v-if="newSource.type === 'csv'">
            <UFormGroup label="File Path">
              <UInput
                v-model="newSource.file_path"
                placeholder="/path/to/data.csv"
              />
            </UFormGroup>
          </template>
          
          <!-- API specific -->
          <template v-if="newSource.type === 'api'">
            <UFormGroup label="API Endpoint">
              <UInput
                v-model="newSource.endpoint"
                placeholder="https://api.example.com/sensors"
              />
            </UFormGroup>
            <UFormGroup label="Polling Interval">
              <USelect
                v-model="newSource.polling_interval"
                :options="pollingOptions"
              />
            </UFormGroup>
          </template>
          
          <!-- IoT Gateway specific -->
          <template v-if="newSource.type === 'iot'">
            <UFormGroup label="Gateway URL">
              <UInput
                v-model="newSource.config.gateway_url"
                placeholder="mqtt://gateway.local:1883"
              />
            </UFormGroup>
          </template>
        </div>
        
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton color="gray" @click="isAddModalOpen = false">
              Cancel
            </UButton>
            <UButton
              color="primary"
              :loading="isSaving"
              @click="saveSource"
            >
              Add Source
            </UButton>
          </div>
        </template>
      </UCard>
    </UModal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Props {
  equipmentId: string
}

const props = defineProps<Props>()

const api = useApiAdvanced()
const toast = useToast()

const isLoading = ref(true)
const isSaving = ref(false)
const isAddModalOpen = ref(false)
const syncingIds = ref<string[]>([])
const dataSources = ref<any[]>([])

const newSource = ref({
  name: '',
  type: '',
  file_path: '',
  endpoint: '',
  polling_interval: '30s',
  config: {
    gateway_url: ''
  }
})

const sourceTypeOptions = [
  { value: 'csv', label: 'CSV File' },
  { value: 'api', label: 'REST API' },
  { value: 'iot', label: 'IoT Gateway' },
  { value: 'simulator', label: 'Data Simulator' }
]

const pollingOptions = [
  { value: '10s', label: 'Every 10 seconds' },
  { value: '30s', label: 'Every 30 seconds' },
  { value: '1m', label: 'Every 1 minute' },
  { value: '5m', label: 'Every 5 minutes' },
  { value: 'manual', label: 'Manual only' }
]

// Load data
async function loadData() {
  isLoading.value = true
  
  try {
    const response = await api.get<any>(
      `/api/data-sources?equipment_id=${props.equipmentId}`
    )
    dataSources.value = response.sources || []
  } catch (error: any) {
    toast.add({
      title: 'Failed to load data sources',
      description: error.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}

// Actions
function openAddSourceModal() {
  newSource.value = {
    name: '',
    type: '',
    file_path: '',
    endpoint: '',
    polling_interval: '30s',
    config: { gateway_url: '' }
  }
  isAddModalOpen.value = true
}

async function saveSource() {
  if (!newSource.value.name || !newSource.value.type) {
    toast.add({
      title: 'Validation error',
      description: 'Please fill all required fields',
      color: 'yellow'
    })
    return
  }
  
  isSaving.value = true
  
  try {
    await api.post('/api/data-sources', {
      equipment_id: props.equipmentId,
      ...newSource.value
    })
    
    toast.add({
      title: 'Data source added',
      description: `${newSource.value.name} has been added`,
      color: 'green'
    })
    
    isAddModalOpen.value = false
    await loadData()
  } catch (error: any) {
    toast.add({
      title: 'Failed to add data source',
      description: error.message,
      color: 'red'
    })
  } finally {
    isSaving.value = false
  }
}

async function syncSource(source: any) {
  syncingIds.value.push(source.id)
  
  try {
    await api.post(`/api/data-sources/${source.id}/sync`)
    
    toast.add({
      title: 'Sync started',
      description: `${source.name} is syncing`,
      color: 'blue'
    })
    
    setTimeout(() => loadData(), 2000)
  } catch (error: any) {
    toast.add({
      title: 'Sync failed',
      description: error.message,
      color: 'red'
    })
  } finally {
    syncingIds.value = syncingIds.value.filter(id => id !== source.id)
  }
}

function editSource(source: any) {
  console.log('Edit source:', source)
  // TODO: Open edit modal
}

async function deleteSource(source: any) {
  const confirmed = confirm(`Delete data source "${source.name}"?`)
  if (!confirmed) return
  
  try {
    await api.delete(`/api/data-sources/${source.id}`)
    
    toast.add({
      title: 'Data source deleted',
      description: `${source.name} has been removed`,
      color: 'green'
    })
    
    await loadData()
  } catch (error: any) {
    toast.add({
      title: 'Failed to delete',
      description: error.message,
      color: 'red'
    })
  }
}

// Helpers
function getSourceIcon(type: string): string {
  const icons: Record<string, string> = {
    csv: 'i-heroicons-document-text',
    api: 'i-heroicons-cloud',
    iot: 'i-heroicons-wifi',
    simulator: 'i-heroicons-beaker'
  }
  return icons[type] || 'i-heroicons-server'
}

function getSourceIconBg(type: string): string {
  const colors: Record<string, string> = {
    csv: 'bg-blue-100 dark:bg-blue-900/30',
    api: 'bg-purple-100 dark:bg-purple-900/30',
    iot: 'bg-green-100 dark:bg-green-900/30',
    simulator: 'bg-yellow-100 dark:bg-yellow-900/30'
  }
  return colors[type] || 'bg-gray-100 dark:bg-gray-800'
}

function getSourceIconColor(type: string): string {
  const colors: Record<string, string> = {
    csv: 'text-blue-600 dark:text-blue-400',
    api: 'text-purple-600 dark:text-purple-400',
    iot: 'text-green-600 dark:text-green-400',
    simulator: 'text-yellow-600 dark:text-yellow-400'
  }
  return colors[type] || 'text-gray-600 dark:text-gray-400'
}

function formatSourceType(type: string): string {
  const labels: Record<string, string> = {
    csv: 'CSV File',
    api: 'REST API',
    iot: 'IoT Gateway',
    simulator: 'Data Simulator'
  }
  return labels[type] || type
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toString()
}

function formatTime(timestamp: string | number | null): string {
  if (!timestamp) return 'Never'
  
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
  return date.toLocaleDateString()
}

// Lifecycle
onMounted(() => {
  loadData()
})
</script>
