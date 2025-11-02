<template>
  <div class="space-y-8">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">{{ t('systems.title') }}</h1>
        <p class="u-body text-gray-600 mt-1">{{ t('systems.subtitle') }}</p>
      </div>
      <button @click="showCreateModal = true" class="u-btn u-btn-primary u-btn-md w-full sm:w-auto">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        {{ t('systems.addNew') }}
      </button>
    </div>

    <div v-if="systems.length > 0" class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      <div v-for="system in systems" :key="system.id" class="u-card p-6 hover:shadow-lg transition-shadow">
        <div class="flex items-start justify-between mb-4">
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-gray-900 truncate">{{ system.name }}</h3>
            <p class="text-sm text-gray-500">{{ system.type }}</p>
          </div>
          <span class="u-badge" :class="getSystemStatusClass(system.status)">
            {{ t(`systems.status.${system.status}`) }}
          </span>
        </div>

        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">{{ t('systems.pressure') }}</span>
            <span class="font-semibold">{{ system.pressure }} бар</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">{{ t('systems.temperature') }}</span>
            <span class="font-semibold">{{ system.temperature }}°C</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">{{ t('systems.updated') }}</span>
            <span class="text-sm text-gray-500">{{ formatDate(system.last_update) }}</span>
          </div>
        </div>

        <div class="flex items-center gap-2 mt-4">
          <NuxtLink :to="`/systems/${system.id}`" class="u-btn u-btn-primary u-btn-sm flex-1">
            <Icon name="heroicons:eye" class="w-4 h-4 mr-1" />
            {{ t('ui.view') }}
          </NuxtLink>
          <button class="u-btn u-btn-ghost u-btn-sm flex-1">
            <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-1" />
            {{ t('ui.settings') }}
          </button>
        </div>
      </div>
    </div>

    <div v-else class="u-card p-12 text-center">
      <div class="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <Icon name="heroicons:server-stack" class="w-8 h-8 text-gray-400" />
      </div>
      <h3 class="text-lg font-semibold text-gray-900 mb-2">{{ t('systems.noSystems') }}</h3>
      <p class="text-gray-600 mb-6">{{ t('systems.noSystemsDesc') }}</p>
      <button @click="showCreateModal = true" class="u-btn u-btn-primary">
        <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
        {{ t('systems.addNew') }}
      </button>
    </div>

    <UCreateSystemModal
      v-model="showCreateModal"
      :loading="isCreating"
      @submit="handleCreateSystem"
      @cancel="showCreateModal = false"
    />
  </div>
</template>

<script setup lang="ts">
import type { HydraulicSystem } from '~/types/api'

// Page metadata
definePageMeta({
  layout: 'dashboard',
  middleware: ['auth']
})

// Composables
const { t } = useI18n()

// State
const showCreateModal = ref(false)
const isCreating = ref(false)

// Mock data
const systems = ref<HydraulicSystem[]>([
  {
    id: 1,
    name: 'HYD-001 - Pump Station A',
    type: 'industrial',
    status: 'active',
    description: 'Основная насосная станция производственной линии',
    pressure: 2.3,
    temperature: 68,
    flow_rate: 185,
    vibration: 0.8,
    health_score: 92,
    last_update: new Date().toISOString(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: 2,
    name: 'HYD-002 - Hydraulic Motor B',
    type: 'mobile',
    status: 'maintenance',
    description: 'Гидравлический мотор мобильной установки',
    pressure: 1.8,
    temperature: 72,
    flow_rate: 150,
    vibration: 1.2,
    health_score: 78,
    last_update: new Date().toISOString(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
])

// Methods
const handleCreateSystem = async (data: any) => {
  isCreating.value = true
  try {
    const newSystem: HydraulicSystem = {
      id: Date.now(),
      name: data.name,
      type: data.type,
      status: data.status || 'active',
      description: data.description,
      pressure: 0,
      temperature: 0,
      flow_rate: 0,
      vibration: 0,
      health_score: 100,
      last_update: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
    systems.value.push(newSystem)
    showCreateModal.value = false
  } catch (error) {
    console.error('Failed to create system:', error)
  } finally {
    isCreating.value = false
  }
}

const getSystemStatusClass = (status: string): string => {
  const classes: Record<string, string> = {
    active: 'u-badge-success',
    maintenance: 'u-badge-warning',
    emergency: 'u-badge-error',
    inactive: 'u-badge-gray'
  }
  return classes[status] || 'u-badge-gray'
}

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}
</script>
