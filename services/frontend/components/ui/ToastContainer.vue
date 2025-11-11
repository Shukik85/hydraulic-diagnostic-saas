<template>
  <Teleport to="body">
    <div class="toast-container">
      <TransitionGroup name="toast">
        <div
          v-for="toast in toasts"
          :key="toast.id"
          :class="toastClasses(toast.type)"
          role="alert"
        >
          <div class="toast-icon">
            <Icon :name="getIcon(toast.type)" :class="getIconColor(toast.type)" />
          </div>
          
          <div class="toast-content">
            <p class="toast-message">{{ toast.message }}</p>
            <p v-if="toast.description" class="toast-description">{{ toast.description }}</p>
          </div>
          
          <button
            v-if="toast.action"
            @click="handleAction(toast)"
            class="toast-action"
          >
            {{ toast.action.label }}
          </button>
          
          <button
            v-if="toast.dismissible"
            @click="dismiss(toast.id)"
            class="toast-close"
            aria-label="Закрыть"
          >
            <Icon name="heroicons:x-mark" />
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { useToast, type Toast, type ToastType } from '~/composables/useToast'

const { toasts, dismiss } = useToast()

const toastClasses = (type: ToastType) => [
  'toast',
  {
    'toast-success': type === 'success',
    'toast-error': type === 'error',
    'toast-warning': type === 'warning',
    'toast-info': type === 'info'
  }
]

function getIcon(type: ToastType): string {
  const icons: Record<ToastType, string> = {
    success: 'heroicons:check-circle',
    error: 'heroicons:x-circle',
    warning: 'heroicons:exclamation-triangle',
    info: 'heroicons:information-circle'
  }
  return icons[type]
}

function getIconColor(type: ToastType): string {
  const colors: Record<ToastType, string> = {
    success: 'text-green-500',
    error: 'text-red-500',
    warning: 'text-orange-500',
    info: 'text-cyan-500'
  }
  return colors[type]
}

async function handleAction(toast: Toast) {
  if (toast.action) {
    try {
      await toast.action.handler()
      dismiss(toast.id)
    } catch (error) {
      console.error('Toast action failed:', error)
    }
  }
}
</script>

<style scoped>
.toast-container {
  @apply fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none;
  max-width: 420px;
}

.toast {
  @apply flex items-start gap-3 p-4 rounded-card shadow-elevated pointer-events-auto;
  @apply bg-white dark:bg-industrial-800;
  @apply border border-industrial-200 dark:border-industrial-700;
  animation: toast-in 0.3s ease-out;
}

.toast-icon {
  @apply flex-shrink-0 w-5 h-5;
}

.toast-content {
  @apply flex-1 min-w-0;
}

.toast-message {
  @apply text-sm font-medium text-industrial-900 dark:text-industrial-50;
}

.toast-description {
  @apply text-xs text-industrial-600 dark:text-industrial-400 mt-1;
}

.toast-action {
  @apply text-sm font-medium text-hydraulic-600 hover:text-hydraulic-700;
  @apply px-2 py-1 rounded transition-colors;
}

.toast-close {
  @apply flex-shrink-0 w-5 h-5 text-industrial-400 hover:text-industrial-600;
  @apply transition-colors;
}

/* Toast type variants */
.toast-success {
  @apply border-l-4 border-l-green-500;
}

.toast-error {
  @apply border-l-4 border-l-red-500;
}

.toast-warning {
  @apply border-l-4 border-l-orange-500;
}

.toast-info {
  @apply border-l-4 border-l-cyan-500;
}

/* Animations */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%) scale(0.95);
}

.toast-move {
  transition: transform 0.3s ease;
}

@keyframes toast-in {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
</style>