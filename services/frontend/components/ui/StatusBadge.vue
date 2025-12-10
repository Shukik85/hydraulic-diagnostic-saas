<!--
  StatusBadge Component
  @component Status indicator badge with accessibility
  @accessibility WCAG 2.1 AA - semantic color + text, proper contrast
-->

<template>
  <span
    class="status-badge"
    :class="statusClass"
    :aria-label="statusLabel"
    :role="ariaRole"
  >
    <span class="status-dot" :aria-hidden="true"></span>
    <span class="status-text">{{ statusDisplay }}</span>
  </span>
</template>

<script setup lang="ts">
import type { SystemStatus } from '~/types/systems'

interface Props {
  status: SystemStatus
  ariaRole?: 'status' | 'img'
}

const props = withDefaults(defineProps<Props>(), {
  ariaRole: 'status',
})

const statusClass = computed(() => {
  const baseClass = 'status-badge'
  const statusMap: Record<SystemStatus, string> = {
    online: `${baseClass}--online`,
    degraded: `${baseClass}--degraded`,
    offline: `${baseClass}--offline`,
  }
  return statusMap[props.status]
})

const statusLabel = computed(() => {
  const labelMap: Record<SystemStatus, string> = {
    online: 'System is online and operating normally',
    degraded: 'System is online but experiencing issues',
    offline: 'System is offline and not operational',
  }
  return labelMap[props.status]
})

const statusDisplay = computed(() => {
  const displayMap: Record<SystemStatus, string> = {
    online: 'Online',
    degraded: 'Degraded',
    offline: 'Offline',
  }
  return displayMap[props.status]
})
</script>

<style scoped lang="css">
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--space-6);
  padding: var(--space-6) var(--space-10);
  border-radius: var(--radius-full);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  white-space: nowrap;
  transition: all var(--duration-fast) var(--ease-standard);
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}

/* Online Status */
.status-badge--online {
  background: rgba(var(--color-success-rgb), 0.15);
  color: var(--color-success);
  border: 1px solid rgba(var(--color-success-rgb), 0.25);
}

.status-badge--online .status-dot {
  background: var(--color-success);
  box-shadow: 0 0 0 3px rgba(var(--color-success-rgb), 0.1);
}

/* Degraded Status */
.status-badge--degraded {
  background: rgba(var(--color-warning-rgb), 0.15);
  color: var(--color-warning);
  border: 1px solid rgba(var(--color-warning-rgb), 0.25);
}

.status-badge--degraded .status-dot {
  background: var(--color-warning);
  box-shadow: 0 0 0 3px rgba(var(--color-warning-rgb), 0.1);
}

/* Offline Status */
.status-badge--offline {
  background: rgba(var(--color-error-rgb), 0.15);
  color: var(--color-error);
  border: 1px solid rgba(var(--color-error-rgb), 0.25);
}

.status-badge--offline .status-dot {
  background: var(--color-error);
  box-shadow: 0 0 0 3px rgba(var(--color-error-rgb), 0.1);
  animation: none;
}

/* Focus styles for accessibility */
.status-badge:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Responsive */
@media (max-width: 768px) {
  .status-badge {
    font-size: var(--font-size-xs);
    padding: var(--space-4) var(--space-8);
    gap: var(--space-4);
  }

  .status-dot {
    width: 6px;
    height: 6px;
  }
}
</style>
