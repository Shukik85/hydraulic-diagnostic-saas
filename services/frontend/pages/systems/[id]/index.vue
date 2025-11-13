<script setup lang="ts">
/**
 * System Detail Page - Type-safe with Generated API
 * 
 * Shows detailed system view with:
 * - System tree visualization
 * - Breadcrumbs navigation
 * - Real-time updates
 * - Component drill-down
 */

import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'
import type { BreadcrumbItem } from '~/components/SystemBreadcrumbs.vue'

definePageMeta({
  middleware: ['auth'],
  layout: 'dashboard'
})

// Route params
const route = useRoute()
const systemId = route.params.id as string

// Composables
const api = useGeneratedApi()
const { success, error: notifyError } = useNotifications()
const sync = useRealtimeSync({ autoConnect: true })

// State
const system = ref<System | null>(null)
const loading = ref(false)

// Breadcrumbs
const breadcrumbs = computed<BreadcrumbItem[]>(() => [
  { label: 'Системы', path: '/systems' },
  { 
    label: system.value?.name || 'Загрузка...', 
    current: true,
    icon: 'heroicons:server'
  }
])

// Load system
async function loadSystem() {
  loading.value = true
  try {
    // ✅ Type-safe API call!
    system.value = await api.equipment.getSystem(systemId)
  } catch (err) {
    notifyError('Ошибка загрузки системы')
    console.error(err)
  } finally {
    loading.value = false
  }
}

// Handle tree navigation
function handleNodeClick({ type, id }: { type: string; id: string }) {
  if (type === 'component') {
    navigateTo(`/systems/${systemId}/components/${id}`)
  } else if (type === 'sensor') {
    navigateTo(`/systems/${systemId}/sensors/${id}`)
  }
}

// Real-time updates
watch(() => sync.lastUpdate, () => {
  if (sync.lastUpdate?.type === 'system_status_update') {
    if (sync.lastUpdate.data.system_id === systemId) {
      loadSystem()
    }
  }
})

// Load on mount
onMounted(() => loadSystem())
</script>

<template>
  <div class="system-detail-page">
    <!-- Breadcrumbs -->
    <SystemBreadcrumbs :items="breadcrumbs" class="mb-6" />

    <ApiState :loading :error="null" :data="system">
      <template #loading>
        <div class="loading-state">
          <Icon name="heroicons:arrow-path" class="w-8 h-8 animate-spin" />
          <p>Загрузка системы...</p>
        </div>
      </template>
      
      <template #default="{ data }">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Left: System info -->
          <div class="lg:col-span-1">
            <div class="info-card">
              <div class="flex items-center gap-3 mb-6">
                <Icon name="heroicons:server" class="w-10 h-10 text-blue-600" />
                <div>
                  <h2 class="text-2xl font-bold text-gray-900 dark:text-white">
                    {{ data.name }}
                  </h2>
                  <span :class="['status-badge-lg', `status-${data.status}`]">
                    {{ data.status }}
                  </span>
                </div>
              </div>
              
              <dl class="info-list">
                <div class="info-row">
                  <dt>Тип оборудования</dt>
                  <dd>{{ data.equipment_type }}</dd>
                </div>
                
                <div class="info-row">
                  <dt>Производитель</dt>
                  <dd>{{ data.manufacturer }}</dd>
                </div>
                
                <div class="info-row">
                  <dt>Модель</dt>
                  <dd>{{ data.model }}</dd>
                </div>
                
                <div class="info-row">
                  <dt>Серийный номер</dt>
                  <dd class="font-mono text-sm">{{ data.serial_number }}</dd>
                </div>
                
                <div v-if="data.location" class="info-row">
                  <dt>Местоположение</dt>
                  <dd>{{ data.location }}</dd>
                </div>
              </dl>
              
              <div class="mt-6 space-y-2">
                <PermissionGate permission="systems:write">
                  <button class="btn-secondary w-full">
                    <Icon name="heroicons:pencil" class="w-4 h-4" />
                    Редактировать
                  </button>
                </PermissionGate>
                
                <button class="btn-secondary w-full">
                  <Icon name="heroicons:chart-bar" class="w-4 h-4" />
                  Аналитика
                </button>
              </div>
            </div>
          </div>
          
          <!-- Right: Tree view -->
          <div class="lg:col-span-2">
            <div class="tree-card">
              <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Структура системы
              </h3>
              
              <SystemTree
                :system="data"
                :expanded-levels="2"
                :show-values="true"
                @node-click="handleNodeClick"
              />
            </div>
          </div>
        </div>
      </template>
    </ApiState>
  </div>
</template>

<style scoped>
.system-detail-page {
  @apply container mx-auto px-4 py-8;
}

.info-card {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6;
}

.tree-card {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6;
}

.info-list {
  @apply space-y-3;
}

.info-row {
  @apply flex justify-between;
}

.info-row dt {
  @apply text-sm text-gray-500;
}

.info-row dd {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.status-badge-lg {
  @apply inline-block px-3 py-1 rounded-full text-sm font-medium;
}

.loading-state {
  @apply flex flex-col items-center justify-center py-20 text-gray-500;
}

.btn-secondary {
  @apply flex items-center justify-center gap-2 px-4 py-2;
  @apply border border-gray-300 dark:border-gray-600 rounded-lg;
  @apply text-gray-700 dark:text-gray-300;
  @apply hover:bg-gray-50 dark:hover:bg-gray-700;
  @apply transition-colors;
}
</style>
