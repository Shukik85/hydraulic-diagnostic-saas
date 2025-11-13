<script setup lang="ts">
/**
 * Systems List Page - Type-safe with Generated API
 * 
 * Shows all hydraulic systems with:
 * - Real-time status updates
 * - Tree view navigation
 * - Advanced filtering
 * - Type-safe API calls
 */

import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

definePageMeta({
  middleware: ['auth'],
  layout: 'dashboard'
})

// Composables
const api = useGeneratedApi()
const { success, error: notifyError } = useNotifications()
const sync = useRealtimeSync({ autoConnect: true })

// State
const systems = ref<System[]>([])
const loading = ref(false)
const searchQuery = ref('')
const statusFilter = ref<string>('all')

// Computed
const filteredSystems = computed(() => {
  let filtered = systems.value
  
  // Search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(s => 
      s.name.toLowerCase().includes(query) ||
      s.manufacturer.toLowerCase().includes(query) ||
      s.model.toLowerCase().includes(query)
    )
  }
  
  // Status filter
  if (statusFilter.value !== 'all') {
    filtered = filtered.filter(s => s.status === statusFilter.value)
  }
  
  return filtered
})

const statusCounts = computed(() => ({
  all: systems.value.length,
  online: systems.value.filter(s => s.status === 'online').length,
  offline: systems.value.filter(s => s.status === 'offline').length,
  maintenance: systems.value.filter(s => s.status === 'maintenance').length,
  error: systems.value.filter(s => s.status === 'error').length
}))

// Load systems
async function loadSystems() {
  loading.value = true
  try {
    // ✅ Type-safe API call!
    systems.value = await api.equipment.getSystems()
    success(`Загружено ${systems.value.length} систем`)
  } catch (err) {
    notifyError('Ошибка загрузки систем')
    console.error(err)
  } finally {
    loading.value = false
  }
}

// Navigate to system details
function viewSystem(systemId: string) {
  navigateTo(`/systems/${systemId}`)
}

// Navigate to create system
function createSystem() {
  navigateTo('/systems/new')
}

// Real-time updates
watch(() => sync.lastUpdate, () => {
  // Reload systems on real-time update
  if (sync.lastUpdate?.type === 'system_status_update') {
    loadSystems()
  }
})

// Load on mount
onMounted(() => loadSystems())
</script>

<template>
  <div class="systems-page">
    <!-- Header -->
    <div class="page-header">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white">
          Гидравлические системы
        </h1>
        <p class="text-gray-600 dark:text-gray-400 mt-1">
          Управление и мониторинг оборудования
        </p>
      </div>
      
      <PermissionGate permission="systems:write">
        <button class="btn-primary" @click="createSystem">
          <Icon name="heroicons:plus" class="w-5 h-5" />
          Создать систему
        </button>
      </PermissionGate>
    </div>

    <!-- Filters -->
    <div class="filters-section">
      <!-- Search -->
      <div class="search-box">
        <Icon name="heroicons:magnifying-glass" class="search-icon" />
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Поиск по названию, производителю или модели..."
          class="search-input"
        />
      </div>
      
      <!-- Status filters -->
      <div class="status-filters">
        <button
          v-for="status in ['all', 'online', 'offline', 'maintenance', 'error']"
          :key="status"
          :class="['status-filter', { active: statusFilter === status }]"
          @click="statusFilter = status"
        >
          {{ status === 'all' ? 'Все' : status }}
          <span class="count">{{ statusCounts[status] }}</span>
        </button>
      </div>
    </div>

    <!-- Loading state -->
    <ApiState :loading :error="null" :data="systems">
      <template #loading>
        <div class="loading-state">
          <Icon name="heroicons:arrow-path" class="w-8 h-8 animate-spin" />
          <p>Загрузка систем...</p>
        </div>
      </template>
      
      <template #default="{ data }">
        <!-- Systems grid -->
        <div v-if="filteredSystems.length > 0" class="systems-grid">
          <div
            v-for="system in filteredSystems"
            :key="system.id"
            class="system-card"
            @click="viewSystem(system.id)"
          >
            <!-- Header -->
            <div class="card-header">
              <Icon name="heroicons:server" class="w-6 h-6 text-blue-600" />
              <div class="flex-1 min-w-0">
                <h3 class="card-title">{{ system.name }}</h3>
                <p class="card-subtitle">
                  {{ system.manufacturer }} {{ system.model }}
                </p>
              </div>
              
              <!-- Status badge -->
              <span :class="['status-badge', `status-${system.status}`]">
                {{ system.status }}
              </span>
            </div>
            
            <!-- Info -->
            <div class="card-info">
              <div class="info-item">
                <Icon name="heroicons:cube" class="w-4 h-4" />
                <span>{{ system.equipment_type }}</span>
              </div>
              
              <div v-if="system.location" class="info-item">
                <Icon name="heroicons:map-pin" class="w-4 h-4" />
                <span>{{ system.location }}</span>
              </div>
              
              <div class="info-item">
                <Icon name="heroicons:cog" class="w-4 h-4" />
                <span>{{ system.components?.length || 0 }} компонентов</span>
              </div>
            </div>
            
            <!-- Actions -->
            <div class="card-actions">
              <button class="btn-sm btn-secondary">
                <Icon name="heroicons:eye" class="w-4 h-4" />
                Просмотр
              </button>
            </div>
          </div>
        </div>
        
        <!-- Empty state -->
        <div v-else class="empty-state">
          <Icon name="heroicons:server" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 class="text-xl font-semibold text-gray-700 mb-2">
            Системы не найдены
          </h3>
          <p class="text-gray-500 mb-6">
            Начните с создания первой системы
          </p>
          <PermissionGate permission="systems:write">
            <button class="btn-primary" @click="createSystem">
              <Icon name="heroicons:plus" class="w-5 h-5" />
              Создать первую систему
            </button>
          </PermissionGate>
        </div>
      </template>
    </ApiState>
  </div>
</template>

<style scoped>
.systems-page {
  @apply container mx-auto px-4 py-8;
}

.page-header {
  @apply flex items-start justify-between mb-8;
}

.filters-section {
  @apply mb-6 space-y-4;
}

.search-box {
  @apply relative;
}

.search-icon {
  @apply absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400;
}

.search-input {
  @apply w-full pl-12 pr-4 py-3 border border-gray-300 dark:border-gray-700 rounded-lg;
  @apply bg-white dark:bg-gray-800 text-gray-900 dark:text-white;
  @apply focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}

.status-filters {
  @apply flex gap-2 overflow-x-auto pb-2;
}

.status-filter {
  @apply px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700;
  @apply text-sm font-medium text-gray-700 dark:text-gray-300;
  @apply hover:bg-gray-50 dark:hover:bg-gray-800;
  @apply transition-colors whitespace-nowrap;
}

.status-filter.active {
  @apply bg-blue-600 text-white border-blue-600;
  @apply hover:bg-blue-700;
}

.status-filter .count {
  @apply ml-2 px-2 py-0.5 rounded-full bg-gray-200 dark:bg-gray-700 text-xs;
}

.status-filter.active .count {
  @apply bg-blue-700;
}

.systems-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6;
}

.system-card {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700;
  @apply hover:border-blue-500 hover:shadow-lg;
  @apply cursor-pointer transition-all p-6;
}

.card-header {
  @apply flex items-start gap-3 mb-4;
}

.card-title {
  @apply font-semibold text-gray-900 dark:text-white truncate;
}

.card-subtitle {
  @apply text-sm text-gray-500 truncate;
}

.status-badge {
  @apply px-2 py-1 rounded-full text-xs font-medium whitespace-nowrap;
}

.status-online {
  @apply bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400;
}

.status-offline {
  @apply bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400;
}

.status-maintenance {
  @apply bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400;
}

.status-error {
  @apply bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400;
}

.card-info {
  @apply space-y-2 mb-4;
}

.info-item {
  @apply flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400;
}

.card-actions {
  @apply flex gap-2 pt-4 border-t border-gray-200 dark:border-gray-700;
}

.loading-state {
  @apply flex flex-col items-center justify-center py-20 text-gray-500;
}

.empty-state {
  @apply text-center py-20;
}

.btn-primary {
  @apply flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg;
  @apply hover:bg-blue-700 transition-colors font-medium;
}

.btn-secondary {
  @apply flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg;
  @apply text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700;
  @apply transition-colors;
}

.btn-sm {
  @apply text-sm px-3 py-1.5;
}
</style>
