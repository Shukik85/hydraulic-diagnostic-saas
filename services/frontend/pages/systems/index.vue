<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('systems.title') }}</h1>
        <p class="text-steel-shine mt-2">{{ t('systems.subtitle') }}</p>
      </div>
      <UButton 
        size="lg"
        @click="showCreateModal = true"
      >
        <Icon name="heroicons:plus" class="w-5 h-5 mr-2" />
        {{ t('systems.addNew') }}
      </UButton>
    </div>
    <!-- Zero State -->
    <UZeroState
      v-if="!loading && systems.length === 0"
      icon-name="heroicons:cube"
      :title="t('systems.empty.title')"
      :description="t('systems.empty.description')"
      action-icon="heroicons:plus"
      :action-text="t('systems.empty.action')"
      @action="showCreateModal = true"
    />
    <!-- Systems Grid -->
    <div v-else class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      <div 
        v-for="system in systems" 
        :key="system.id" 
        class="card-interactive p-6"
        role="button"
        tabindex="0"
        @click="navigateTo(`/systems/${system.id}`)"
        @keydown.enter="navigateTo(`/systems/${system.id}`)"
      >
        <!-- Header with Status -->
        <div class="flex items-start justify-between mb-4">
          <div class="flex-1 min-w-0">
            <h3 class="font-semibold text-white truncate text-lg">{{ system.name }}</h3>
            <p class="text-sm text-steel-shine">{{ t(`systems.types.${system.type}`) }}</p>
          </div>
          <UStatusDot 
            :status="getSystemStatusType(system.status)"
            :label="t(`systems.status.${system.status}`)"
          />
        </div>
        <!-- Metrics -->
        <div class="space-y-3 mb-4">
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:arrow-trending-up" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">{{ t('systems.pressure') }}</span>
            </div>
            <span class="font-semibold text-white">{{ system.pressure }} бар</span>
          </div>
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:fire" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">{{ t('systems.temperature') }}</span>
            </div>
            <span class="font-semibold text-white">{{ system.temperature }}°C</span>
          </div>
          <div class="flex justify-between items-center">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:heart" class="w-4 h-4 text-steel-400" />
              <span class="text-sm text-steel-shine">Здоровье</span>
            </div>
            <div class="flex items-center gap-2">
              <div class="w-16 progress-bar">
                <div 
                  :class="getHealthColorClass(system.health_score)"
                  :style="{ width: system.health_score + '%' }"
                />
              </div>
              <span class="text-sm font-semibold text-white">{{ system.health_score }}%</span>
            </div>
          </div>
        </div>
        <!-- Footer -->
        <div class="flex items-center justify-between pt-4 border-t border-steel-700/50">
          <div class="flex items-center gap-1.5 text-xs text-steel-400">
            <Icon name="heroicons:clock" class="w-3 h-3" />
            <span>{{ formatDate(system.last_update) }}</span>
          </div>
          <div class="flex items-center gap-2">
            <button 
              class="btn-icon"
              @click.stop="handleSettings(system.id)"
              aria-label="Настройки"
            >
              <Icon name="heroicons:cog-6-tooth" class="w-5 h-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>
    <!-- Create System Modal -->
    <UCreateSystemModal
      v-model="showCreateModal"
      :loading="isCreating"
      @submit="handleCreateSystem"
    />
  </div>
</template>
<script setup lang="ts">
import type { HydraulicSystem } from '~/types/api'
import { useSeoMeta } from '#imports'

definePageMeta({
  layout: 'dashboard' as const,
  middleware: ['auth']
})

const { t } = useI18n()

useSeoMeta({
  title: 'Системы | Hydraulic Diagnostic SaaS',
  description: 'Список и мониторинг всех подключённых гидравлических систем предприятия. AI-анализ и статус в реальном времени.',
  ogTitle: 'Systems | Hydraulic Diagnostic SaaS',
  ogDescription: 'Hydraulic systems monitoring, status, and real-time anomaly analysis',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

const showCreateModal = ref(false)
const isCreating = ref(false)
const loading = ref(false)

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
  },
  {
    id: 3,
    name: 'HYD-003 - Control Valve C',
    type: 'construction',
    status: 'active',
    description: 'Управляющий клапан строительного оборудования',
    pressure: 2.1,
    temperature: 65,
    flow_rate: 120,
    vibration: 0.5,
    health_score: 95,
    last_update: new Date().toISOString(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
])

const handleCreateSystem = async (data: any): Promise<void> => {
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

const handleSettings = (systemId: number): void => {
  navigateTo(`/systems/${systemId}/settings`)
}

const getSystemStatusType = (status: string): 'success' | 'warning' | 'error' | 'offline' => {
  const statusMap: Record<string, 'success' | 'warning' | 'error' | 'offline'> = {
    active: 'success',
    maintenance: 'warning',
    emergency: 'error',
    inactive: 'offline'
  }
  return statusMap[status] || 'offline'
}

const getHealthColorClass = (score: number): string => {
  if (score >= 90) return 'progress-fill-success'
  if (score >= 70) return 'progress-fill-warning'
  return 'progress-fill-error'
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
