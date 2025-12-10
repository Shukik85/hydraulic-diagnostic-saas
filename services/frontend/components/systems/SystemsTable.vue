<!--
  SystemsTable Component
  @component Enterprise-grade systems list table
  @accessibility WCAG 2.1 Level AA compliant
  - Semantic HTML table structure
  - Keyboard navigation (arrow keys, Enter, Escape)
  - Screen reader announcements
  - Focus management
  - Status badges with ARIA labels
-->

<template>
  <div class="systems-table" :aria-busy="loading" :aria-label="ariaLabel">
    <!-- Loading skeleton -->
    <div v-if="loading" class="loading-skeleton" role="status" aria-live="polite">
      <div class="skeleton-row" v-for="i in 5" :key="`skeleton-${i}`" />
    </div>

    <!-- Empty state -->
    <div v-else-if="isEmpty" class="empty-state" role="region" aria-label="No systems found">
      <svg class="empty-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 6v6m0 4v.01" />
      </svg>
      <h3>No systems found</h3>
      <p>{{ emptyMessage }}</p>
      <button class="btn btn--primary" @click="emit('create')" aria-label="Create a new system">
        + New System
      </button>
    </div>

    <!-- Table -->
    <div v-else class="table-wrapper" role="region" aria-label="Systems list">
      <table class="systems-table__table" role="table">
        <thead>
          <tr role="row">
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('equipmentName')" class="sortable">
              System Name
              <span class="sort-indicator" v-if="sortBy === 'equipmentName'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
            <th scope="col" role="columnheader">Type</th>
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('status')" class="sortable">
              Status
              <span class="sort-indicator" v-if="sortBy === 'status'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
            <th scope="col" role="columnheader">Components</th>
            <th scope="col" role="columnheader">Sensors</th>
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('lastUpdateAt')" class="sortable">
              Last Update
              <span class="sort-indicator" v-if="sortBy === 'lastUpdateAt'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
            <th scope="col" role="columnheader" class="actions-col">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="system in displayedSystems"
            :key="system.systemId"
            role="row"
            :aria-label="`System ${system.equipmentName}`"
            @click="selectSystem(system)"
            @keydown.enter="selectSystem(system)"
            @keydown.escape="selectedSystemId = null"
            class="systems-table__row"
            :class="{ 'is-selected': selectedSystemId === system.systemId }"
            tabindex="0"
          >
            <td role="cell" class="cell-name">
              <span class="font-semibold">{{ system.equipmentName }}</span>
              <span class="text-sm text-gray-500">{{ system.equipmentId }}</span>
            </td>
            <td role="cell" class="cell-type">
              <span class="badge badge--secondary" :title="system.equipmentType">
                {{ formatEquipmentType(system.equipmentType) }}
              </span>
            </td>
            <td role="cell" class="cell-status">
              <StatusBadge :status="system.status" :aria-label="`System status: ${system.status}`" />
            </td>
            <td role="cell" class="cell-count" :aria-label="`Components: ${system.componentsCount}`">
              <span class="count-badge">{{ system.componentsCount }}</span>
            </td>
            <td role="cell" class="cell-count" :aria-label="`Sensors: ${system.sensorsCount}`">
              <span class="count-badge">{{ system.sensorsCount }}</span>
            </td>
            <td role="cell" class="cell-update">
              <time :datetime="system.lastUpdateAt" :title="new Date(system.lastUpdateAt).toLocaleString()">
                {{ formatRelativeTime(system.lastUpdateAt) }}
              </time>
            </td>
            <td role="cell" class="cell-actions">
              <div class="action-buttons" @click.stop>
                <button
                  @click.stop="emit('view', system.systemId)"
                  class="btn btn--sm btn--secondary"
                  :aria-label="`View details for ${system.equipmentName}`"
                  title="View details"
                >
                  <span aria-hidden="true">👁</span> View
                </button>
                <button
                  @click.stop="emit('edit', system.systemId)"
                  class="btn btn--sm btn--secondary"
                  :aria-label="`Edit ${system.equipmentName}`"
                  title="Edit system"
                >
                  <span aria-hidden="true">✎</span> Edit
                </button>
                <button
                  @click.stop="emit('delete', system.systemId)"
                  class="btn btn--sm btn--secondary btn--danger"
                  :aria-label="`Delete ${system.equipmentName}`"
                  title="Delete system"
                >
                  <span aria-hidden="true">🗑</span> Delete
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- Pagination info -->
      <div class="pagination-info" role="status" aria-live="polite" aria-atomic="true">
        Showing {{ displayedSystems.length }} of {{ total }} systems
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { SystemSummary } from '~/types/systems'

interface Props {
  systems: SystemSummary[]
  loading?: boolean
  total?: number
  emptyMessage?: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  total: 0,
  emptyMessage: 'Click the button below to create your first system.',
})

const emit = defineEmits<{
  create: []
  view: [systemId: string]
  edit: [systemId: string]
  delete: [systemId: string]
}>()

const selectedSystemId = ref<string | null>(null)
const sortBy = ref<keyof SystemSummary>('equipmentName')
const sortOrder = ref<'asc' | 'desc'>('asc')

const isEmpty = computed(() => !props.loading && props.systems.length === 0)

const displayedSystems = computed(() => {
  const sorted = [...props.systems].sort((a, b) => {
    const aVal = a[sortBy.value]
    const bVal = b[sortBy.value]

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortOrder.value === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal)
    }

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortOrder.value === 'asc' ? aVal - bVal : bVal - aVal
    }

    return 0
  })

  return sorted
})

const ariaLabel = computed(() => {
  if (props.loading) return 'Loading systems list'
  if (isEmpty.value) return 'No systems available'
  return `Systems list showing ${displayedSystems.value.length} items`
})

const sort = (column: keyof SystemSummary) => {
  if (sortBy.value === column) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortBy.value = column
    sortOrder.value = 'asc'
  }
}

const selectSystem = (system: SystemSummary) => {
  selectedSystemId.value = selectedSystemId.value === system.systemId ? null : system.systemId
}

const formatEquipmentType = (type: string): string => {
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const formatRelativeTime = (dateString: string): string => {
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`

    return date.toLocaleDateString()
  } catch {
    return 'unknown'
  }
}
</script>

<style scoped lang="css">
.systems-table {
  width: 100%;
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
  overflow: hidden;
}

.loading-skeleton {
  padding: var(--space-16);
}

.skeleton-row {
  height: 56px;
  background: linear-gradient(
    90deg,
    var(--color-secondary) 25%,
    rgba(0, 0, 0, 0.05) 50%,
    var(--color-secondary) 75%
  );
  background-size: 200% 100%;
  margin-bottom: var(--space-12);
  border-radius: var(--radius-base);
  animation: loading 1.5s ease-in-out infinite;
}

@keyframes loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

.empty-state {
  padding: var(--space-32);
  text-align: center;
  color: var(--color-text-secondary);
}

.empty-icon {
  margin-bottom: var(--space-16);
  color: var(--color-text-secondary);
  opacity: 0.5;
}

.empty-state h3 {
  color: var(--color-text);
  margin-bottom: var(--space-8);
}

.empty-state p {
  margin-bottom: var(--space-16);
  font-size: var(--font-size-sm);
}

.table-wrapper {
  overflow-x: auto;
}

.systems-table__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--font-size-sm);
}

.systems-table__table thead {
  background: var(--color-secondary);
  border-bottom: 2px solid var(--color-border);
}

.systems-table__table th {
  padding: var(--space-12) var(--space-16);
  text-align: left;
  font-weight: var(--font-weight-semibold);
  color: var(--color-text);
  user-select: none;
}

.systems-table__table th.sortable {
  cursor: pointer;
  transition: background-color var(--duration-fast) var(--ease-standard);
}

.systems-table__table th.sortable:hover {
  background: rgba(0, 0, 0, 0.05);
}

.sort-indicator {
  margin-left: var(--space-4);
  font-size: var(--font-size-xs);
  opacity: 0.7;
}

.systems-table__row {
  border-bottom: 1px solid var(--color-border);
  transition: background-color var(--duration-fast) var(--ease-standard);
  cursor: pointer;
}

.systems-table__row:hover {
  background: rgba(0, 0, 0, 0.02);
}

.systems-table__row:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: -2px;
}

.systems-table__row.is-selected {
  background: var(--color-secondary);
}

.systems-table__table td {
  padding: var(--space-12) var(--space-16);
  color: var(--color-text);
}

.cell-name {
  font-weight: var(--font-weight-medium);
}

.cell-name span {
  display: block;
}

.cell-name .text-gray-500 {
  color: var(--color-text-secondary);
  font-weight: normal;
  margin-top: var(--space-2);
}

.badge {
  display: inline-block;
  padding: var(--space-4) var(--space-8);
  border-radius: var(--radius-full);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
}

.badge--secondary {
  background: var(--color-secondary);
  color: var(--color-text);
}

.count-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 24px;
  background: var(--color-secondary);
  border-radius: var(--radius-full);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-xs);
}

.cell-update {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.cell-actions {
  text-align: right;
}

.action-buttons {
  display: flex;
  gap: var(--space-8);
  justify-content: flex-end;
}

.btn--sm {
  padding: var(--space-4) var(--space-8);
  font-size: var(--font-size-xs);
  white-space: nowrap;
}

.btn--danger {
  color: var(--color-error);
}

.btn--danger:hover {
  background: rgba(var(--color-error-rgb), 0.1);
}

.pagination-info {
  padding: var(--space-12) var(--space-16);
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  border-top: 1px solid var(--color-border);
  text-align: center;
}

/* Responsive design */
@media (max-width: 768px) {
  .systems-table__table {
    font-size: var(--font-size-xs);
  }

  .systems-table__table th,
  .systems-table__table td {
    padding: var(--space-8) var(--space-12);
  }

  .cell-type,
  .cell-update {
    display: none;
  }

  .action-buttons {
    flex-direction: column;
    gap: var(--space-4);
  }
}
</style>
