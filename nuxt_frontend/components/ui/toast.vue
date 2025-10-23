<template>
  <TransitionGroup
    name="toast"
    tag="div"
    class="fixed top-4 right-4 z-50 space-y-2 max-w-sm"
  >
    <div
      v-for="toast in toasts"
      :key="toast.id"
      :class="[
        'p-4 rounded-lg shadow-lg border transition-all duration-300 transform',
        toast.type === 'success' && 'bg-green-50 border-green-200 text-green-800',
        toast.type === 'error' && 'bg-red-50 border-red-200 text-red-800',
        toast.type === 'warning' && 'bg-yellow-50 border-yellow-200 text-yellow-800',
        toast.type === 'info' && 'bg-blue-50 border-blue-200 text-blue-800'
      ]"
    >
      <div class="flex items-start gap-3">
        <Icon
          :name="getIcon(toast.type)"
          class="h-5 w-5 mt-0.5 flex-shrink-0"
        />
        <div class="flex-1">
          <p class="text-sm font-medium">{{ toast.title }}</p>
          <p v-if="toast.description" class="text-sm opacity-90 mt-1">
            {{ toast.description }}
          </p>
        </div>
        <UiButton
          variant="ghost"
          size="icon"
          class="h-6 w-6 flex-shrink-0 hover:bg-black/10"
          @click="removeToast(toast.id)"
        >
          <Icon name="lucide:x" class="h-4 w-4" />
        </UiButton>
      </div>
    </div>
  </TransitionGroup>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

interface Toast {
  id: string
  title: string
  description?: string
  type: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

const toasts = ref<Toast[]>([])
let toastCounter = 0

const getIcon = (type: string) => {
  const icons = {
    success: 'lucide:check-circle',
    error: 'lucide:x-circle',
    warning: 'lucide:alert-triangle',
    info: 'lucide:info'
  }
  return icons[type as keyof typeof icons] || 'lucide:info'
}

const addToast = (toast: Omit<Toast, 'id'>) => {
  const id = `toast-${++toastCounter}`
  const newToast: Toast = {
    id,
    duration: 5000,
    ...toast
  }

  toasts.value.push(newToast)

  if (newToast.duration && newToast.duration > 0) {
    setTimeout(() => {
      removeToast(id)
    }, newToast.duration)
  }
}

const removeToast = (id: string) => {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

// Expose methods for global use
defineExpose({
  addToast,
  removeToast
})

// Global toast function
if (typeof window !== 'undefined') {
  (window as any).$toast = {
    success: (title: string, description?: string) => addToast({ title, description, type: 'success' }),
    error: (title: string, description?: string) => addToast({ title, description, type: 'error' }),
    warning: (title: string, description?: string) => addToast({ title, description, type: 'warning' }),
    info: (title: string, description?: string) => addToast({ title, description, type: 'info' })
  }
}
</script>

<style scoped>
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
  transform: translateX(100%);
}
</style>
