<template>
  <div class="equipment-list-page space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between flex-wrap gap-4">
      <div>
        <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Equipment List
        </h1>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Manage your hydraulic equipment
        </p>
      </div>
      
      <div class="flex gap-2 items-center flex-wrap">
        <UInput
          v-model="search"
          placeholder="Search equipment..."
          icon="i-heroicons-magnifying-glass"
          class="w-64"
        />
        <USelect
          v-model="typeFilter"
          :options="equipmentTypes"
          placeholder="Type"
          class="w-40"
        />
        <USelect
          v-model="statusFilter"
          :options="statusOptions"
          placeholder="Status"
          class="w-32"
        />
        <UButton
          color="primary"
          icon="i-heroicons-plus"
          @click="addEquipment"
        >
          Add Equipment
        </UButton>
      </div>
    </div>
    
    <!-- Loading state -->
    <div v-if="isLoading" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      <USkeleton v-for="i in 6" :key="i" class="h-40 w-full rounded-lg" />
    </div>
    
    <!-- Empty state -->
    <UCard v-else-if="filteredEquipment.length === 0" class="p-12">
      <div class="text-center">
        <UIcon
          name="i-heroicons-archive-box"
          class="w-16 h-16 text-gray-400 dark:text-gray-600 mx-auto mb-4"
        />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
          No equipment found
        </h3>
        <p class="text-sm text-gray-600 dark:text-gray-400 mb-6">
          {{ search ? 'Try adjusting your search filters' : 'Get started by adding your first equipment' }}
        </p>
        <UButton
          v-if="!search"
          color="primary"
          icon="i-heroicons-plus"
          @click="addEquipment"
        >
          Add Equipment
        </UButton>
      </div>
    </UCard>
    
    <!-- Equipment cards -->
    <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      <UCard
        v-for="equip in filteredEquipment"
        :key="equip.id"
        class="p-6 hover:shadow-lg transition-shadow cursor-pointer"
        @click="viewEquipment(equip.id)"
      >
        <!-- Header -->
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
              <UIcon
                name="i-heroicons-cpu-chip"
                class="w-6 h-6 text-blue-600 dark:text-blue-400"
              />
            </div>
            <div>
              <h3 class="font-semibold text-gray-900 dark:text-gray-100">
                {{ equip.name || equip.model || equip.id }}
              </h3>
              <p class="text-xs text-gray-500 dark:text-gray-400">
                {{ equip.equipment_type }}
              </p>
            </div>
          </div>
          <UBadge
            :color="getStatusColor(equip.status)"
            variant="soft"
          >
            {{ getStatusLabel(equip.status) }}
          </UBadge>
        </div>
        
        <!-- Info -->
        <div class="space-y-1 mb-4 text-sm">
          <div v-if="equip.manufacturer" class="text-gray-600 dark:text-gray-400">
            <span class="font-medium">Manufacturer:</span> {{ equip.manufacturer }}
          </div>
          <div v-if="equip.model" class="text-gray-600 dark:text-gray-400">
            <span class="font-medium">Model:</span> {{ equip.model }}
          </div>
        </div>
        
        <!-- Stats -->
        <div class="grid grid-cols-3 gap-2 mb-4">
          <div class="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
            <p class="text-xs text-gray-500 dark:text-gray-400">Sensors</p>
            <p class="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {{ equip.sensor_count || 0 }}
            </p>
          </div>
          <div class="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
            <p class="text-xs text-gray-500 dark:text-gray-400">Uptime</p>
            <p class="text-lg font-semibold text-green-600 dark:text-green-400">
              {{ equip.uptime || 0 }}%
            </p>
          </div>
          <div class="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
            <p class="text-xs text-gray-500 dark:text-gray-400">Alerts</p>
            <p class="text-lg font-semibold text-red-600 dark:text-red-400">
              {{ equip.alert_count || 0 }}
            </p>
          </div>
        </div>
        
        <!-- Actions -->
        <div class="flex gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
          <UButton
            size="sm"
            color="primary"
            class="flex-1"
            @click.stop="viewEquipment(equip.id)"
          >
            View
          </UButton>
          <UButton
            size="sm"
            color="gray"
            variant="outline"
            icon="i-heroicons-pencil"
            @click.stop="editEquipment(equip.id)"
          />
          <UButton
            size="sm"
            color="red"
            variant="ghost"
            icon="i-heroicons-trash"
            @click.stop="deleteEquipment(equip.id)"
          />
        </div>
      </UCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

definePageMeta({
  layout: 'dashboard',
  middleware: ['auth']
})

const api = useApiAdvanced()
const toast = useToast()
const router = useRouter()

const search = ref('')
const typeFilter = ref('')
const statusFilter = ref('')
const equipmentList = ref<any[]>([])
const isLoading = ref(true)

const equipmentTypes = [
  { label: 'All Types', value: '' },
  { label: 'Excavator', value: 'excavator' },
  { label: 'Press', value: 'press' },
  { label: 'Crane', value: 'crane' },
  { label: 'Loader', value: 'loader' },
  { label: 'Pump', value: 'pump' }
]

const statusOptions = [
  { label: 'All Statuses', value: '' },
  { label: 'Active', value: 'active' },
  { label: 'Inactive', value: 'inactive' },
  { label: 'Maintenance', value: 'maintenance' }
]

// Fetch equipment
async function fetchEquipment() {
  isLoading.value = true
  
  try {
    const response = await api.get<any>('/api/equipment')
    equipmentList.value = response.equipment || []
  } catch (error: any) {
    toast.add({
      title: 'Failed to load equipment',
      description: error.message,
      color: 'red'
    })
  } finally {
    isLoading.value = false
  }
}

// Filtered equipment
const filteredEquipment = computed(() => {
  return equipmentList.value.filter(e => {
    const matchesSearch = !search.value ||
      (e.name || e.model || '').toLowerCase().includes(search.value.toLowerCase())
    const matchesType = !typeFilter.value || e.equipment_type === typeFilter.value
    const matchesStatus = !statusFilter.value || e.status === statusFilter.value
    
    return matchesSearch && matchesType && matchesStatus
  })
})

// Actions
function addEquipment() {
  router.push('/system-metadata/wizard')
}

function viewEquipment(id: string) {
  router.push(`/equipment/${id}`)
}

function editEquipment(id: string) {
  router.push(`/equipment/${id}?tab=settings`)
}

async function deleteEquipment(id: string) {
  const equip = equipmentList.value.find(e => e.id === id)
  const confirmed = confirm(`Delete equipment "${equip?.name || id}"? This cannot be undone.`)
  if (!confirmed) return
  
  try {
    await api.delete(`/api/equipment/${id}`)
    
    toast.add({
      title: 'Equipment deleted',
      description: 'Equipment has been permanently deleted',
      color: 'green'
    })
    
    await fetchEquipment()
  } catch (error: any) {
    toast.add({
      title: 'Failed to delete equipment',
      description: error.message,
      color: 'red'
    })
  }
}

// Helpers
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    active: 'green',
    inactive: 'gray',
    maintenance: 'yellow'
  }
  return colors[status] || 'gray'
}

function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    active: 'Active',
    inactive: 'Inactive',
    maintenance: 'Maintenance'
  }
  return labels[status] || 'Unknown'
}

// Lifecycle
onMounted(() => {
  fetchEquipment()
})
</script>
