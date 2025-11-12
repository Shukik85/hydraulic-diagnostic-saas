<!--
  EquipmentSensors.vue — Таб с датчиками оборудования
  
  Features:
  - Таблица всех сенсоров с метаданными
  - Статус каждого сенсора (online/offline)
  - Последние показания
  - Mapping status (mapped/unmapped)
  - Quick actions (view, edit, delete)
  - Add sensor button
-->
<template>
  <div class="equipment-sensors space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Sensors
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
          {{ sensors.length }} sensors configured
        </p>
      </div>
      
      <UButton
        color="primary"
        icon="i-heroicons-plus"
        @click="openAddSensorModal"
      >
        Add Sensor
      </UButton>
    </div>
    
    <!-- Loading state -->
    <div v-if="isLoading" class="space-y-4">
      <USkeleton class="h-12 w-full" />
      <USkeleton v-for="i in 5" :key="i" class="h-16 w-full" />
    </div>
    
    <!-- Empty state -->
    <UCard v-else-if="sensors.length === 0" class="p-12">
      <div class="text-center">
        <UIcon
          name="i-heroicons-signal-slash"
          class="w-16 h-16 text-gray-400 dark:text-gray-600 mx-auto mb-4"
        />
        <h4 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          No sensors configured
        </h4>
        <p class="text-sm text-gray-600 dark:text-gray-400 mb-6">
          Add sensors to start monitoring this equipment
        </p>
        <UButton
          color="primary"
          icon="i-heroicons-plus"
          @click="openAddSensorModal"
        >
          Add First Sensor
        </UButton>
      </div>
    </UCard>
    
    <!-- Sensors table -->
    <UCard v-else class="overflow-hidden">
      <UTable
        :rows="sensors"
        :columns="columns"
        :loading="isLoading"
      >
        <!-- Sensor name with icon -->
        <template #name-data="{ row }">
          <div class="flex items-center gap-3">
            <div
              class="w-8 h-8 rounded-lg flex items-center justify-center"
              :class="getSensorIconBg(row.type)"
            >
              <UIcon
                :name="getSensorIcon(row.type)"
                class="w-4 h-4"
                :class="getSensorIconColor(row.type)"
              />
            </div>
            <div>
              <p class="font-medium text-gray-900 dark:text-gray-100">
                {{ row.name }}
              </p>
              <p class="text-xs text-gray-500 dark:text-gray-400">
                {{ row.sensor_id }}
              </p>
            </div>
          </div>
        </template>
        
        <!-- Sensor type -->
        <template #type-data="{ row }">
          <span class="text-sm text-gray-700 dark:text-gray-300">
            {{ formatSensorType(row.type) }}
          </span>
        </template>
        
        <!-- Status badge -->
        <template #status-data="{ row }">
          <UBadge
            :color="row.status === 'online' ? 'green' : 'gray'"
            variant="soft"
          >
            <div class="flex items-center gap-1">
              <div
                class="w-1.5 h-1.5 rounded-full"
                :class="row.status === 'online' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'"
              />
              {{ row.status === 'online' ? 'Online' : 'Offline' }}
            </div>
          </UBadge>
        </template>
        
        <!-- Last reading -->
        <template #lastReading-data="{ row }">
          <div v-if="row.lastReading">
            <p class="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {{ row.lastReading.value }} {{ row.unit }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              {{ formatTime(row.lastReading.timestamp) }}
            </p>
          </div>
          <span v-else class="text-sm text-gray-400 dark:text-gray-600">
            No data
          </span>
        </template>
        
        <!-- Mapping status -->
        <template #mapping-data="{ row }">
          <UBadge
            :color="row.component_id ? 'blue' : 'yellow'"
            variant="soft"
          >
            {{ row.component_id ? 'Mapped' : 'Unmapped' }}
          </UBadge>
        </template>
        
        <!-- Actions -->
        <template #actions-data="{ row }">
          <div class="flex items-center gap-2">
            <UButton
              size="xs"
              color="gray"
              variant="ghost"
              icon="i-heroicons-chart-bar"
              @click="viewSensorData(row)"
            />
            <UButton
              size="xs"
              color="gray"
              variant="ghost"
              icon="i-heroicons-pencil"
              @click="editSensor(row)"
            />
            <UButton
              size="xs"
              color="red"
              variant="ghost"
              icon="i-heroicons-trash"
              @click="deleteSensor(row)"
            />
          </div>
        </template>
      </UTable>
    </UCard>
    
    <!-- Add Sensor Modal -->
    <UModal v-model="isAddModalOpen" :ui="{ width: 'sm:max-w-2xl' }">
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">
              Add Sensor
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
          <UFormGroup label="Sensor Name" required>
            <UInput
              v-model="newSensor.name"
              placeholder="e.g., Pressure Sensor A"
            />
          </UFormGroup>
          
          <UFormGroup label="Sensor ID" required>
            <UInput
              v-model="newSensor.sensor_id"
              placeholder="e.g., PS-001"
            />
          </UFormGroup>
          
          <UFormGroup label="Sensor Type" required>
            <USelect
              v-model="newSensor.type"
              :options="sensorTypeOptions"
            />
          </UFormGroup>
          
          <UFormGroup label="Unit">
            <UInput
              v-model="newSensor.unit"
              placeholder="e.g., bar, °C, L/min"
            />
          </UFormGroup>
          
          <UFormGroup label="Component Mapping">
            <USelect
              v-model="newSensor.component_id"
              :options="componentOptions"
              placeholder="Select component (optional)"
            />
          </UFormGroup>
        </div>
        
        <template #footer>
          <div class="flex justify-end gap-3">
            <UButton color="gray" @click="isAddModalOpen = false">
              Cancel
            </UButton>
            <UButton
              color="primary"
              :loading="isSaving"
              @click="saveSensor"
            >
              Add Sensor
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
const sensors = ref<any[]>([])
const components = ref<any[]>([])

const newSensor = ref({
  name: '',
  sensor_id: '',
  type: '',
  unit: 'bar',
  component_id: null as string | null
})

const columns = [
  { key: 'name', label: 'Sensor' },
  { key: 'type', label: 'Type' },
  { key: 'status', label: 'Status' },
  { key: 'lastReading', label: 'Last Reading' },
  { key: 'mapping', label: 'Mapping' },
  { key: 'actions', label: '' }
]

const sensorTypeOptions = [
  { value: 'pressure', label: 'Pressure' },
  { value: 'temperature', label: 'Temperature' },
  { value: 'flow', label: 'Flow Rate' },
  { value: 'level', label: 'Level' },
  { value: 'vibration', label: 'Vibration' },
  { value: 'position', label: 'Position' }
]

const componentOptions = ref<Array<{ value: string; label: string }>>([])

// Load data
async function loadData() {
  isLoading.value = true
  
  try {
    // Load sensors
    const sensorsResponse = await api.get<any>(
      `/api/sensor-mappings?equipment_id=${props.equipmentId}`
    )
    sensors.value = sensorsResponse.mappings || []
    
    // Load components for mapping dropdown
    const componentsResponse = await api.get<any>(
      `/api/metadata/systems?equipment_id=${props.equipmentId}`
    )
    components.value = componentsResponse.system?.components || []
    
    // Build component options
    componentOptions.value = components.value.map(c => ({
      value: c.id,
      label: c.name
    }))
    
  } catch (error: any) {
    toast.add({
      title: 'Failed to load sensors',
      description: error.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}

// Actions
function openAddSensorModal() {
  newSensor.value = {
    name: '',
    sensor_id: '',
    type: '',
    unit: 'bar',
    component_id: null
  }
  isAddModalOpen.value = true
}

async function saveSensor() {
  if (!newSensor.value.name || !newSensor.value.sensor_id || !newSensor.value.type) {
    toast.add({
      title: 'Validation error',
      description: 'Please fill all required fields',
      color: 'yellow'
    })
    return
  }
  
  isSaving.value = true
  
  try {
    await api.post('/api/sensor-mappings', {
      equipment_id: props.equipmentId,
      ...newSensor.value
    })
    
    toast.add({
      title: 'Sensor added',
      description: `${newSensor.value.name} has been added`,
      color: 'green'
    })
    
    isAddModalOpen.value = false
    await loadData()
  } catch (error: any) {
    toast.add({
      title: 'Failed to add sensor',
      description: error.message,
      color: 'red'
    })
  } finally {
    isSaving.value = false
  }
}

function viewSensorData(sensor: any) {
  navigateTo(`/sensors/${sensor.id}`)
}

function editSensor(sensor: any) {
  console.log('Edit sensor:', sensor)
  // TODO: Open edit modal
}

async function deleteSensor(sensor: any) {
  const confirmed = confirm(`Delete sensor "${sensor.name}"?`)
  if (!confirmed) return
  
  try {
    await api.delete(`/api/sensor-mappings/${sensor.id}`)
    
    toast.add({
      title: 'Sensor deleted',
      description: `${sensor.name} has been removed`,
      color: 'green'
    })
    
    await loadData()
  } catch (error: any) {
    toast.add({
      title: 'Failed to delete sensor',
      description: error.message,
      color: 'red'
    })
  }
}

// Helpers
function getSensorIcon(type: string): string {
  const icons: Record<string, string> = {
    pressure: 'i-heroicons-arrow-trending-up',
    temperature: 'i-heroicons-fire',
    flow: 'i-heroicons-beaker',
    level: 'i-heroicons-signal',
    vibration: 'i-heroicons-bolt',
    position: 'i-heroicons-map-pin'
  }
  return icons[type] || 'i-heroicons-cpu-chip'
}

function getSensorIconBg(type: string): string {
  const colors: Record<string, string> = {
    pressure: 'bg-blue-100 dark:bg-blue-900/30',
    temperature: 'bg-red-100 dark:bg-red-900/30',
    flow: 'bg-green-100 dark:bg-green-900/30',
    level: 'bg-purple-100 dark:bg-purple-900/30',
    vibration: 'bg-yellow-100 dark:bg-yellow-900/30',
    position: 'bg-indigo-100 dark:bg-indigo-900/30'
  }
  return colors[type] || 'bg-gray-100 dark:bg-gray-800'
}

function getSensorIconColor(type: string): string {
  const colors: Record<string, string> = {
    pressure: 'text-blue-600 dark:text-blue-400',
    temperature: 'text-red-600 dark:text-red-400',
    flow: 'text-green-600 dark:text-green-400',
    level: 'text-purple-600 dark:text-purple-400',
    vibration: 'text-yellow-600 dark:text-yellow-400',
    position: 'text-indigo-600 dark:text-indigo-400'
  }
  return colors[type] || 'text-gray-600 dark:text-gray-400'
}

function formatSensorType(type: string): string {
  return type.charAt(0).toUpperCase() + type.slice(1)
}

function formatTime(timestamp: string | number): string {
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
