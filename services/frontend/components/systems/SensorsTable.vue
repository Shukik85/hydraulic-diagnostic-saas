<!--
  Sensors Table Component
  @component Real-time sensor readings table
  @accessibility WCAG 2.1 AA - semantic structure, live region updates
-->

<template>
  <div class="sensors-table" :aria-busy="loading" :aria-label="ariaLabel">
    <!-- Loading skeleton -->
    <div v-if="loading" class="loading-state" role="status" aria-live="polite">
      <div class="skeleton-row" v-for="i in 5" :key="`skeleton-${i}`" />
    </div>

    <!-- Empty state -->
    <div v-else-if="isEmpty" class="empty-state" role="region" aria-label="No sensors">
      <svg class="empty-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="16" />
        <line x1="8" y1="12" x2="16" y2="12" />
      </svg>
      <h3>No sensors configured</h3>
      <p>This system does not have any sensors configured yet.</p>
    </div>

    <!-- Table -->
    <div v-else class="table-wrapper" role="region" aria-label="Sensor readings">
      <table class="sensors-table__table" role="table">
        <thead>
          <tr role="row">
            <th scope="col" role="columnheader">Sensor ID</th>
            <th scope="col" role="columnheader">Type</th>
            <th scope="col" role="columnheader">Component</th>
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('lastValue')" class="sortable">
              Value
              <span class="sort-indicator" v-if="sortBy === 'lastValue'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
            <th scope="col" role="columnheader">Unit</th>
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('status')" class="sortable">
              Status
              <span class="sort-indicator" v-if="sortBy === 'status'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
            <th scope="col" role="columnheader" aria-sort="none" @click="sort('lastUpdateAt')" class="sortable">
              Last Update
              <span class="sort-indicator" v-if="sortBy === 'lastUpdateAt'" :aria-label="`Sorted ${sortOrder}`">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="sensor in displayedSensors"
            :key="sensor.sensorId"
            role="row"
            :aria-label="`${sensor.sensorType} sensor ${sensor.sensorId} reading ${sensor.lastValue}${sensor.unit}`"
            class="sensors-table__row"
            :class="getRowClass(sensor)"
          >
            <td role="cell" class="cell-id">
              <code>{{ sensor.sensorId }}</code>
            </td>
            <td role="cell" class="cell-type">
              <span class="badge badge--secondary">{{ formatSensorType(sensor.sensorType) }}</span>
            </td>
            <td role="cell" class="cell-component">
              {{ sensor.componentId }}
            </td>
            <td role="cell" class="cell-value">
              <span class="value-display" :class="getValueClass(sensor)" :aria-label="getValueLabel(sensor)">
                {{ formatValue(sensor.lastValue, sensor.sensorType) }}
              </span>
              <span v-if="sensor.normalRange" class="value-range">
                ({{ sensor.normalRange[0] }} - {{ sensor.normalRange[1] }})
              </span>
            </td>
            <td role="cell" class="cell-unit">
              {{ sensor.unit }}
            </td>
            <td role="cell" class="cell-status">
              <StatusBadge :status="sensor.status" :aria-label="`Sensor status: ${sensor.status}`" />
            </td>
            <td role="cell" class="cell-update">
              <time
                :datetime="sensor.lastUpdateAt"
                :title="new Date(sensor.lastUpdateAt).toLocaleString()"
                :aria-label="`Updated ${formatRelativeTime(sensor.lastUpdateAt)}`"
              >
                {{ formatRelativeTime(sensor.lastUpdateAt) }}
              </time>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- Update notice -->
      <div class="update-notice" role="status" aria-live="polite" aria-atomic="true">
        <span v-if="lastUpdateTime" class="update-time">
          Updated {{ lastUpdateTime }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { SystemSensor, SensorStatus } from '~/types/systems'

interface Props {
  sensors: SystemSensor[]
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
})

const sortBy = ref<keyof SystemSensor>('sensorId')
const sortOrder = ref<'asc' | 'desc'>('asc')
const lastUpdateTime = ref<string>('')

const isEmpty = computed(() => !props.loading && props.sensors.length === 0)

const displayedSensors = computed(() => {
  const sorted = [...props.sensors].sort((a, b) => {
    const aVal = a[sortBy.value]
    const bVal = b[sortBy.value]

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      const comparison = aVal.localeCompare(bVal as string)
      return sortOrder.value === 'asc' ? comparison : -comparison
    }

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortOrder.value === 'asc' ? aVal - bVal : bVal - aVal
    }

    return 0
  })

  return sorted
})

const ariaLabel = computed(() => {
  if (props.loading) return 'Loading sensor data'
  if (isEmpty.value) return 'No sensors configured'
  return `Showing ${displayedSensors.value.length} sensors`
})

const sort = (column: keyof SystemSensor) => {
  if (sortBy.value === column) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortBy.value = column
    sortOrder.value = 'asc'
  }
}

const formatSensorType = (type: string): string => {
  const typeMap: Record<string, string> = {
    pressure: 'Pressure',
    temperature: 'Temperature',
    vibration: 'Vibration',
    rpm: 'RPM',
    position: 'Position',
    flow_rate: 'Flow Rate',
  }
  return typeMap[type] || type
}

const formatValue = (value: number | string, type: string): string => {
  if (typeof value !== 'number') return String(value)

  // Format numeric values with appropriate precision
  if (type === 'temperature') {
    return value.toFixed(1)
  }
  if (type === 'pressure') {
    return value.toFixed(2)
  }
  if (type === 'vibration') {
    return value.toFixed(3)
  }

  return value.toString()
}

const getValueClass = (sensor: SystemSensor): string => {
  const classes = ['value-display']

  if (sensor.isError) {
    classes.push('value-display--error')
  } else if (sensor.isWarning) {
    classes.push('value-display--warning')
  }

  return classes.join(' ')
}

const getValueLabel = (sensor: SystemSensor): string => {
  let label = `${sensor.lastValue}${sensor.unit}`

  if (sensor.isError) {
    label += ', error status'
  } else if (sensor.isWarning) {
    label += ', warning status'
  }

  return label
}

const getRowClass = (sensor: SystemSensor): Record<string, boolean> => ({
  'is-error': sensor.status === 'error',
  'is-warning': sensor.status === 'warning',
  'is-offline': sensor.status === 'offline',
})

const formatRelativeTime = (dateString: string): string => {
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffSecs = Math.floor(diffMs / 1000)
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)

    if (diffSecs < 30) return 'now'
    if (diffMins < 1) return `${diffSecs}s ago`
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`

    return date.toLocaleDateString()
  } catch {
    return 'unknown'
  }
}

// Update last update time
watch(
  () => props.sensors,
  () => {
    if (props.sensors.length > 0) {
      const latest = props.sensors.reduce((prev, current) =>
        new Date(current.lastUpdateAt) > new Date(prev.lastUpdateAt) ? current : prev
      )
      lastUpdateTime.value = formatRelativeTime(latest.lastUpdateAt)
    }
  },
  { immediate: true }
)
</script>

<style scoped lang="css">
.sensors-table {
  width: 100%;
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
  overflow: hidden;
}

.loading-state {
  padding: var(--space-16);
}

.skeleton-row {
  height: 48px;
  background: linear-gradient(
    90deg,
    var(--color-secondary) 25%,
    rgba(0, 0, 0, 0.05) 50%,
    var(--color-secondary) 75%
  );
  background-size: 200% 100%;
  margin-bottom: var(--space-8);
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

.table-wrapper {
  overflow-x: auto;
}

.sensors-table__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--font-size-sm);
}

.sensors-table__table thead {
  background: var(--color-secondary);
  border-bottom: 2px solid var(--color-border);
}

.sensors-table__table th {
  padding: var(--space-12) var(--space-16);
  text-align: left;
  font-weight: var(--font-weight-semibold);
  color: var(--color-text);
  user-select: none;
}

.sensors-table__table th.sortable {
  cursor: pointer;
  transition: background-color var(--duration-fast) var(--ease-standard);
}

.sensors-table__table th.sortable:hover {
  background: rgba(0, 0, 0, 0.05);
}

.sort-indicator {
  margin-left: var(--space-4);
  font-size: var(--font-size-xs);
  opacity: 0.7;
}

.sensors-table__row {
  border-bottom: 1px solid var(--color-border);
  transition: background-color var(--duration-fast) var(--ease-standard);
}

.sensors-table__row:hover {
  background: rgba(0, 0, 0, 0.02);
}

.sensors-table__row.is-error {
  background: rgba(var(--color-error-rgb), 0.05);
}

.sensors-table__row.is-warning {
  background: rgba(var(--color-warning-rgb), 0.05);
}

.sensors-table__table td {
  padding: var(--space-12) var(--space-16);
  color: var(--color-text);
  vertical-align: middle;
}

.cell-id {
  font-family: var(--font-family-mono);
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.cell-id code {
  background: var(--color-secondary);
  padding: var(--space-2) var(--space-6);
  border-radius: var(--radius-sm);
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

.cell-value {
  font-weight: var(--font-weight-semibold);
  font-size: var(--font-size-base);
}

.value-display {
  display: block;
  padding: var(--space-4) 0;
}

.value-display--error {
  color: var(--color-error);
  font-weight: var(--font-weight-bold);
}

.value-display--warning {
  color: var(--color-warning);
  font-weight: var(--font-weight-bold);
}

.value-range {
  display: block;
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  font-weight: normal;
  margin-top: var(--space-2);
}

.cell-update {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.update-notice {
  padding: var(--space-8) var(--space-16);
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
  border-top: 1px solid var(--color-border);
  background: rgba(var(--color-success-rgb), 0.05);
}

.update-time {
  display: inline-flex;
  align-items: center;
  gap: var(--space-4);
}

.update-time::before {
  content: '•';
  animation: blink 1s ease-in-out infinite;
}

@keyframes blink {
  0%,
  49%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}

/* Responsive */
@media (max-width: 768px) {
  .sensors-table__table {
    font-size: var(--font-size-xs);
  }

  .sensors-table__table th,
  .sensors-table__table td {
    padding: var(--space-8) var(--space-12);
  }

  .cell-component,
  .cell-unit,
  .cell-update {
    display: none;
  }

  .cell-value {
    font-size: var(--font-size-sm);
  }
}
</style>
