<!--
  System Details Page
  @page /systems/[id]
  @description Display complete system information with topology and sensors
  @accessibility WCAG 2.1 AA - tab navigation, real-time updates, keyboard support
-->

<template>
  <div class="system-details-page">
    <!-- Loading state -->
    <div v-if="loading" class="loading-container" role="status" aria-live="polite">
      <div class="spinner" aria-hidden="true"></div>
      <p>Loading system details...</p>
    </div>

    <!-- Error state -->
    <div v-else-if="error" class="error-container" role="alert">
      <div class="error-message">
        <span aria-hidden="true">⚠️</span>
        <p>{{ error }}</p>
      </div>
      <button class="btn btn--secondary" @click="goBack" aria-label="Go back to systems list">
        Back to Systems
      </button>
    </div>

    <!-- Content -->
    <div v-else-if="system" class="system-content">
      <!-- Header -->
      <div class="system-header">
        <div class="header-top">
          <button
            class="btn btn--secondary btn--icon"
            @click="goBack"
            aria-label="Back to systems list"
          >
            ← Back
          </button>
          <div class="header-actions">
            <button
              class="btn btn--secondary"
              @click="navigateTo(`/systems/${system.systemId}/edit`)"
              aria-label="Edit system"
            >
              ✏️ Edit
            </button>
            <button
              class="btn btn--secondary btn--danger"
              @click="confirmDelete"
              aria-label="Delete system"
            >
              🗑️ Delete
            </button>
          </div>
        </div>

        <div class="header-main">
          <div class="header-info">
            <h1>{{ system.equipmentName }}</h1>
            <p class="equipment-id">{{ system.equipmentId }}</p>
          </div>
          <StatusBadge :status="system.status" />
        </div>

        <div class="header-meta">
          <div class="meta-item">
            <span class="meta-label">Type:</span>
            <span class="meta-value">{{ formatEquipmentType(system.equipmentType) }}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Last Update:</span>
            <span class="meta-value">{{ formatLastUpdate(system.lastUpdateAt) }}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Topology Version:</span>
            <span class="meta-value">{{ system.topologyVersion }}</span>
          </div>
        </div>
      </div>

      <!-- Stats -->
      <div class="stats-bar" role="region" aria-label="System statistics">
        <div class="stat">
          <div class="stat-value">{{ system.componentsCount }}</div>
          <div class="stat-label">Components</div>
        </div>
        <div class="stat">
          <div class="stat-value">{{ system.sensorsCount }}</div>
          <div class="stat-label">Sensors</div>
        </div>
        <div class="stat">
          <div class="stat-value">{{ system.operatingHours }}</div>
          <div class="stat-label">Operating Hours</div>
        </div>
      </div>

      <!-- Tabs -->
      <div class="tabs-container">
        <div class="tabs" role="tablist">
          <button
            v-for="tab in tabs"
            :key="tab.id"
            :id="`tab-${tab.id}`"
            role="tab"
            :aria-selected="activeTab === tab.id"
            :aria-controls="`panel-${tab.id}`"
            :tabindex="activeTab === tab.id ? 0 : -1"
            @click="activeTab = tab.id"
            @keydown.arrow-left="handleTabNavigation('left')"
            @keydown.arrow-right="handleTabNavigation('right')"
            @keydown.home="activeTab = tabs[0].id"
            @keydown.end="activeTab = tabs[tabs.length - 1].id"
            class="tab"
            :class="{ 'tab--active': activeTab === tab.id }"
          >
            {{ tab.label }}
          </button>
        </div>

        <!-- Tab Panels -->
        <div class="tab-panels">
          <!-- Overview Panel -->
          <div
            v-show="activeTab === 'overview'"
            :id="`panel-overview`"
            role="tabpanel"
            :aria-labelledby="`tab-overview`"
            class="tab-panel"
          >
            <div class="section">
              <h2>Overview</h2>
              <div class="overview-grid">
                <div class="overview-item">
                  <span class="label">Manufacturer:</span>
                  <span class="value">{{ system.manufacturer || 'N/A' }}</span>
                </div>
                <div class="overview-item">
                  <span class="label">Serial Number:</span>
                  <span class="value">{{ system.serialNumber || 'N/A' }}</span>
                </div>
                <div class="overview-item">
                  <span class="label">Created:</span>
                  <span class="value" :title="new Date(system.createdAt).toLocaleString()">
                    {{ formatDate(system.createdAt) }}
                  </span>
                </div>
                <div class="overview-item">
                  <span class="label">Updated:</span>
                  <span class="value" :title="new Date(system.updatedAt).toLocaleString()">
                    {{ formatDate(system.updatedAt) }}
                  </span>
                </div>
              </div>

              <div v-if="system.description" class="description-section">
                <h3>Description</h3>
                <p>{{ system.description }}</p>
              </div>
            </div>
          </div>

          <!-- Topology Panel -->
          <div
            v-show="activeTab === 'topology'"
            :id="`panel-topology`"
            role="tabpanel"
            :aria-labelledby="`tab-topology`"
            class="tab-panel"
          >
            <div class="section">
              <h2>System Topology</h2>
              <div class="topology-info">
                <div class="topology-section">
                  <h3>Components ({{ system.components.length }})</h3>
                  <div v-if="system.components.length" class="components-list">
                    <div v-for="comp in system.components" :key="comp.componentId" class="component-item">
                      <span class="component-type">{{ comp.componentType }}</span>
                      <span class="component-name">{{ comp.name }}</span>
                      <span class="component-location">{{ comp.location }}</span>
                      <StatusBadge :status="comp.status" />
                    </div>
                  </div>
                  <p v-else class="empty-text">No components defined</p>
                </div>

                <div class="topology-section">
                  <h3>Connections ({{ system.edges.length }})</h3>
                  <div v-if="system.edges.length" class="edges-list">
                    <div v-for="edge in system.edges" :key="edge.edgeId" class="edge-item">
                      <span class="edge-type">{{ edge.edgeType }}</span>
                      <span class="edge-material" v-if="edge.material">{{ edge.material }}</span>
                    </div>
                  </div>
                  <p v-else class="empty-text">No connections defined</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Sensors Panel -->
          <div
            v-show="activeTab === 'sensors'"
            :id="`panel-sensors`"
            role="tabpanel"
            :aria-labelledby="`tab-sensors`"
            class="tab-panel"
          >
            <div class="section">
              <div class="sensors-header">
                <h2>Real-time Sensors</h2>
                <div class="sensors-status" role="status" aria-live="polite">
                  <span v-if="sensorsConnected" class="connection-badge connection-badge--online">
                    ● Live
                  </span>
                  <span v-else class="connection-badge connection-badge--offline">
                    ● Polling
                  </span>
                </div>
              </div>

              <div v-if="sensorsSummary" class="sensors-summary">
                <div class="summary-item summary-item--ok">
                  <span class="count">{{ sensorsSummary.ok }}</span>
                  <span class="label">OK</span>
                </div>
                <div class="summary-item summary-item--warning">
                  <span class="count">{{ sensorsSummary.warning }}</span>
                  <span class="label">Warning</span>
                </div>
                <div class="summary-item summary-item--error">
                  <span class="count">{{ sensorsSummary.error }}</span>
                  <span class="label">Error</span>
                </div>
                <div class="summary-item summary-item--offline">
                  <span class="count">{{ sensorsSummary.offline }}</span>
                  <span class="label">Offline</span>
                </div>
              </div>

              <SensorsTable
                :sensors="sensorsList"
                :loading="sensorsLoading"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Delete Modal -->
    <DeleteConfirmModal
      v-if="showDeleteModal"
      :system-name="system?.equipmentName || ''"
      @confirm="deleteSystem"
      @cancel="showDeleteModal = false"
    />
  </div>
</template>

<script setup lang="ts">
import type { SystemDetail } from '~/types/systems'

definePageMeta({
  middleware: 'auth',
  layout: 'default',
})

const route = useRoute()
const router = useRouter()
const toast = useToast()

const systemId = route.params.id as string

const { fetchSystemById, deleteSystem: deleteSystemApi } = useSystems()
const { sensors, isConnected, isLoading, sensorsSummary } = useSensorData({
  systemId,
  pollingInterval: 5000,
  autoConnect: false, // Will start after system loads
})

const system = ref<SystemDetail | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const showDeleteModal = ref(false)
const activeTab = ref<'overview' | 'topology' | 'sensors'>('overview')

const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'topology', label: 'Topology' },
  { id: 'sensors', label: 'Sensors' },
]

const sensorsConnected = computed(() => isConnected.value)
const sensorsLoading = computed(() => isLoading.value)
const sensorsList = computed(() => sensors.value)

const formatEquipmentType = (type: string): string => {
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const formatLastUpdate = (date: string): string => {
  try {
    const now = new Date()
    const diff = now.getTime() - new Date(date).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 1) return 'just now'
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    return new Date(date).toLocaleDateString()
  } catch {
    return 'unknown'
  }
}

const formatDate = (date: string): string => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const handleTabNavigation = (direction: 'left' | 'right') => {
  const currentIndex = tabs.findIndex((t) => t.id === activeTab.value)
  let nextIndex = direction === 'left' ? currentIndex - 1 : currentIndex + 1

  if (nextIndex < 0) nextIndex = tabs.length - 1
  if (nextIndex >= tabs.length) nextIndex = 0

  activeTab.value = tabs[nextIndex].id
}

const goBack = () => {
  router.back()
}

const confirmDelete = () => {
  showDeleteModal.value = true
}

const deleteSystem = async () => {
  if (!system.value) return

  try {
    await deleteSystemApi(system.value.systemId)
    toast?.success('System deleted successfully')
    navigateTo('/systems')
  } catch (err) {
    toast?.error('Failed to delete system')
  }
}

// Load system details
onMounted(async () => {
  try {
    loading.value = true
    error.value = null
    await fetchSystemById(systemId)
    // Get the system from composable state
    const { currentSystem } = useSystems()
    watch(
      currentSystem,
      (newSystem) => {
        if (newSystem) {
          system.value = newSystem
        }
      },
      { immediate: true }
    )
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load system'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped lang="css">
.system-details-page {
  padding: var(--space-24);
  max-width: 1200px;
  margin: 0 auto;
}

.loading-container,
.error-container {
  text-align: center;
  padding: var(--space-32);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--color-secondary);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto var(--space-16);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-message {
  background: rgba(var(--color-error-rgb), 0.1);
  border: 1px solid var(--color-error);
  border-radius: var(--radius-lg);
  padding: var(--space-16);
  color: var(--color-error);
  margin-bottom: var(--space-16);
}

/* Header */
.system-header {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-24);
  margin-bottom: var(--space-24);
}

.header-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-16);
}

.header-actions {
  display: flex;
  gap: var(--space-12);
}

.header-main {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--space-20);
  gap: var(--space-16);
}

.header-info h1 {
  font-size: var(--font-size-4xl);
  margin-bottom: var(--space-8);
}

.equipment-id {
  color: var(--color-text-secondary);
  font-size: var(--font-size-lg);
}

.header-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-16);
  padding-top: var(--space-16);
  border-top: 1px solid var(--color-border);
}

.meta-item {
  display: flex;
  gap: var(--space-8);
}

.meta-label {
  font-weight: var(--font-weight-semibold);
  color: var(--color-text);
}

.meta-value {
  color: var(--color-text-secondary);
}

/* Stats Bar */
.stats-bar {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-16);
  margin-bottom: var(--space-24);
}

.stat {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-16);
  text-align: center;
}

.stat-value {
  font-size: var(--font-size-4xl);
  font-weight: var(--font-weight-bold);
  color: var(--color-primary);
  margin-bottom: var(--space-8);
}

.stat-label {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

/* Tabs */
.tabs {
  display: flex;
  gap: 0;
  border-bottom: 2px solid var(--color-border);
  background: var(--color-surface);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  padding: 0 var(--space-16);
}

.tab {
  background: none;
  border: none;
  padding: var(--space-12) var(--space-16);
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-medium);
  color: var(--color-text-secondary);
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: all var(--duration-fast) var(--ease-standard);
  margin-bottom: -2px;
}

.tab:hover {
  color: var(--color-text);
}

.tab--active {
  color: var(--color-primary);
  border-bottom-color: var(--color-primary);
}

.tab:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: -2px;
}

.tab-panels {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-top: none;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

.tab-panel {
  padding: var(--space-24);
  animation: fadeIn 150ms ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.section h2 {
  font-size: var(--font-size-2xl);
  margin-bottom: var(--space-16);
}

.section h3 {
  font-size: var(--font-size-lg);
  margin-bottom: var(--space-12);
  margin-top: var(--space-16);
}

/* Overview Grid */
.overview-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-16);
  margin-bottom: var(--space-24);
}

.overview-item {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.label {
  font-weight: var(--font-weight-semibold);
  color: var(--color-text);
  font-size: var(--font-size-sm);
}

.value {
  color: var(--color-text-secondary);
}

.description-section {
  background: var(--color-secondary);
  padding: var(--space-16);
  border-radius: var(--radius-base);
}

.description-section p {
  margin: 0;
  line-height: var(--line-height-normal);
}

/* Topology */
.topology-info {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-24);
}

.topology-section {
  display: flex;
  flex-direction: column;
}

.components-list,
.edges-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-8);
}

.component-item,
.edge-item {
  display: flex;
  align-items: center;
  gap: var(--space-12);
  padding: var(--space-12);
  background: var(--color-secondary);
  border-radius: var(--radius-base);
}

.component-type,
.edge-type {
  background: var(--color-primary);
  color: white;
  padding: var(--space-4) var(--space-8);
  border-radius: var(--radius-full);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  white-space: nowrap;
}

.component-name {
  font-weight: var(--font-weight-semibold);
  flex: 1;
}

.component-location {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
}

.edge-material {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  margin-left: auto;
}

.empty-text {
  color: var(--color-text-secondary);
  font-style: italic;
}

/* Sensors */
.sensors-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-16);
}

.connection-badge {
  padding: var(--space-6) var(--space-12);
  border-radius: var(--radius-full);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
}

.connection-badge--online {
  background: rgba(var(--color-success-rgb), 0.15);
  color: var(--color-success);
}

.connection-badge--offline {
  background: rgba(var(--color-warning-rgb), 0.15);
  color: var(--color-warning);
}

.sensors-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-12);
  margin-bottom: var(--space-24);
}

.summary-item {
  background: var(--color-secondary);
  padding: var(--space-12);
  border-radius: var(--radius-base);
  text-align: center;
  border-left: 4px solid transparent;
}

.summary-item--ok {
  border-left-color: var(--color-success);
}

.summary-item--warning {
  border-left-color: var(--color-warning);
}

.summary-item--error {
  border-left-color: var(--color-error);
}

.summary-item--offline {
  border-left-color: var(--color-text-secondary);
}

.count {
  display: block;
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-bold);
  margin-bottom: var(--space-4);
}

.summary-item .label {
  font-size: var(--font-size-xs);
}

/* Responsive */
@media (max-width: 1024px) {
  .header-main {
    flex-direction: column;
  }

  .topology-info,
  .overview-grid {
    grid-template-columns: 1fr;
  }

  .stats-bar {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .system-details-page {
    padding: var(--space-12);
  }

  .header-top {
    flex-direction: column;
    gap: var(--space-12);
  }

  .header-actions {
    width: 100%;
  }

  .header-actions button {
    flex: 1;
  }

  .stats-bar {
    grid-template-columns: 1fr;
  }

  .sensors-summary {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
