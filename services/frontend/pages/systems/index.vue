<!--
  Systems List Page
  @page /systems
  @description Enterprise systems management dashboard
  @accessibility WCAG 2.1 AA - semantic structure, keyboard nav, screen reader optimized
-->

<template>
  <div class="systems-page">
    <!-- Page header -->
    <div class="page-header">
      <div class="header-content">
        <h1>Systems Dashboard</h1>
        <p class="subtitle">Manage and monitor your hydraulic systems</p>
      </div>
      <button
        class="btn btn--primary btn--lg"
        @click="navigateTo('/systems/create')"
        aria-label="Create a new system"
      >
        <span aria-hidden="true">+</span> New System
      </button>
    </div>

    <!-- Stats cards -->
    <div class="stats-grid" role="region" aria-label="Systems statistics">
      <div class="stat-card">
        <div class="stat-value">{{ statsData.total }}</div>
        <div class="stat-label">Total Systems</div>
        <div class="stat-subtext">Across all equipment</div>
      </div>
      <div class="stat-card stat-card--success">
        <div class="stat-value">{{ statsData.online }}</div>
        <div class="stat-label">Online</div>
        <div class="stat-subtext">{{ onlinePercentage }}% operational</div>
      </div>
      <div class="stat-card stat-card--warning">
        <div class="stat-value">{{ statsData.degraded }}</div>
        <div class="stat-label">Degraded</div>
        <div class="stat-subtext">Require attention</div>
      </div>
      <div class="stat-card stat-card--error">
        <div class="stat-value">{{ statsData.offline }}</div>
        <div class="stat-label">Offline</div>
        <div class="stat-subtext">Not operational</div>
      </div>
    </div>

    <!-- Filters bar -->
    <div class="filters-bar" role="region" aria-label="Search and filter options">
      <div class="filter-group">
        <label for="search-input" class="form-label">Search systems</label>
        <div class="search-wrapper">
          <input
            id="search-input"
            v-model="searchQuery"
            type="text"
            placeholder="Search by name, ID, or type..."
            class="form-control search-input"
            aria-label="Search systems by name, ID, or type"
            @input="debounceSearch"
          />
          <span class="search-icon" aria-hidden="true">🔍</span>
        </div>
      </div>

      <div class="filter-group">
        <label for="status-filter" class="form-label">Status</label>
        <select
          id="status-filter"
          v-model="selectedStatus"
          class="form-control"
          aria-label="Filter by system status"
        >
          <option value="">All Statuses</option>
          <option value="online">Online</option>
          <option value="degraded">Degraded</option>
          <option value="offline">Offline</option>
        </select>
      </div>

      <div class="filter-group">
        <label for="type-filter" class="form-label">Equipment Type</label>
        <select
          id="type-filter"
          v-model="selectedType"
          class="form-control"
          aria-label="Filter by equipment type"
        >
          <option value="">All Types</option>
          <option value="excavator">Excavator</option>
          <option value="loader">Loader</option>
          <option value="dozer">Dozer</option>
          <option value="roller">Roller</option>
        </select>
      </div>

      <div class="filter-group filter-group--actions">
        <button
          class="btn btn--secondary"
          @click="resetFilters"
          aria-label="Clear all filters and search"
        >
          Clear Filters
        </button>
      </div>
    </div>

    <!-- Systems table -->
    <div class="table-container">
      <SystemsTable
        :systems="filteredSystems"
        :loading="loading"
        :total="pagination.total"
        @create="navigateTo('/systems/create')"
        @view="viewSystem"
        @edit="editSystem"
        @delete="confirmDelete"
      />
    </div>

    <!-- Delete confirmation modal -->
    <DeleteConfirmModal
      v-if="showDeleteModal"
      :system-name="systemToDelete?.equipmentName || ''"
      @confirm="deleteSystem"
      @cancel="showDeleteModal = false"
    />
  </div>
</template>

<script setup lang="ts">
import { debounce } from 'lodash-es'
import type { SystemSummary } from '~/types/systems'

definePageMeta({
  middleware: 'auth',
  layout: 'default',
})

const router = useRouter()
const { $fetch } = useNuxtApp()
const toast = useToast()

const { systems, loading, pagination, fetchSystems, deleteSystem: deleteSystemApi } = useSystems()

const searchQuery = ref('')
const selectedStatus = ref('')
const selectedType = ref('')
const showDeleteModal = ref(false)
const systemToDelete = ref<SystemSummary | null>(null)

const statsData = computed(() => {
  const online = systems.value.filter((s) => s.status === 'online').length
  const degraded = systems.value.filter((s) => s.status === 'degraded').length
  const offline = systems.value.filter((s) => s.status === 'offline').length

  return {
    total: systems.value.length,
    online,
    degraded,
    offline,
  }
})

const onlinePercentage = computed(() => {
  if (statsData.value.total === 0) return 0
  return Math.round((statsData.value.online / statsData.value.total) * 100)
})

const filteredSystems = computed(() => {
  let filtered = [...systems.value]

  // Search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(
      (s) =>
        s.equipmentName.toLowerCase().includes(query) ||
        s.equipmentId.toLowerCase().includes(query) ||
        s.equipmentType.toLowerCase().includes(query)
    )
  }

  // Status filter
  if (selectedStatus.value) {
    filtered = filtered.filter((s) => s.status === selectedStatus.value)
  }

  // Type filter
  if (selectedType.value) {
    filtered = filtered.filter((s) => s.equipmentType === selectedType.value)
  }

  return filtered
})

const debounceSearch = debounce(() => {
  // Filters are reactive, so just the debounce prevents excessive re-renders
}, 300)

const resetFilters = () => {
  searchQuery.value = ''
  selectedStatus.value = ''
  selectedType.value = ''
}

const viewSystem = (systemId: string) => {
  navigateTo(`/systems/${systemId}`)
}

const editSystem = (systemId: string) => {
  navigateTo(`/systems/${systemId}/edit`)
}

const confirmDelete = (system: SystemSummary) => {
  systemToDelete.value = system
  showDeleteModal.value = true
}

const deleteSystem = async () => {
  if (!systemToDelete.value) return

  try {
    await deleteSystemApi(systemToDelete.value.systemId)
    showDeleteModal.value = false
    systemToDelete.value = null
    toast?.success('System deleted successfully')
  } catch (err) {
    toast?.error('Failed to delete system')
  }
}

// Load systems on mount
onMounted(async () => {
  await fetchSystems()
})
</script>

<style scoped lang="css">
.systems-page {
  padding: var(--space-24);
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--space-32);
  gap: var(--space-16);
}

.header-content h1 {
  font-size: var(--font-size-4xl);
  margin-bottom: var(--space-8);
  color: var(--color-text);
}

.subtitle {
  color: var(--color-text-secondary);
  font-size: var(--font-size-lg);
}

.btn--lg {
  padding: var(--space-12) var(--space-24);
  font-size: var(--font-size-base);
  white-space: nowrap;
  align-self: flex-start;
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-16);
  margin-bottom: var(--space-32);
}

.stat-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-20);
  transition: all var(--duration-normal) var(--ease-standard);
}

.stat-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--color-primary);
}

.stat-value {
  font-size: var(--font-size-4xl);
  font-weight: var(--font-weight-bold);
  color: var(--color-text);
  margin-bottom: var(--space-8);
}

.stat-label {
  font-weight: var(--font-weight-semibold);
  color: var(--color-text);
  margin-bottom: var(--space-4);
}

.stat-subtext {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.stat-card--success .stat-value {
  color: var(--color-success);
}

.stat-card--warning .stat-value {
  color: var(--color-warning);
}

.stat-card--error .stat-value {
  color: var(--color-error);
}

/* Filters Bar */
.filters-bar {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-20);
  margin-bottom: var(--space-24);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--space-16);
}

.filter-group {
  display: flex;
  flex-direction: column;
}

.filter-group--actions {
  display: flex;
  align-items: flex-end;
  gap: var(--space-8);
}

.form-label {
  display: block;
  margin-bottom: var(--space-8);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
  color: var(--color-text);
}

.search-wrapper {
  position: relative;
}

.search-input {
  padding-left: var(--space-32);
}

.search-icon {
  position: absolute;
  left: var(--space-10);
  top: 50%;
  transform: translateY(-50%);
  opacity: 0.5;
  pointer-events: none;
}

.table-container {
  margin-top: var(--space-24);
}

/* Responsive */
@media (max-width: 1024px) {
  .systems-page {
    padding: var(--space-16);
  }

  .page-header {
    flex-direction: column;
    align-items: stretch;
  }

  .btn--lg {
    width: 100%;
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .filters-bar {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
