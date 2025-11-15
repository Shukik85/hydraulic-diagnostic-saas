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
        'bg-steel-darker backdrop-blur-sm',
        // Variant borders and accents
        toast.type === 'success' && 'border-success-500/40 shadow-[0_0_12px_rgba(34,197,94,0.2)]',
        toast.type === 'error' && 'border-error-500/40 shadow-[0_0_12px_rgba(239,68,68,0.2)]',
        toast.type === 'warning' && 'border-warning-500/40 shadow-[0_0_12px_rgba(251,191,36,0.2)]',
        toast.type === 'info' && 'border-primary-500/40 shadow-[0_0_12px_rgba(79,70,229,0.2)]',
      ]"
    >
      <div class="flex items-start gap-3">
        <!-- Icon with variant color -->
        <Icon 
          :name="getIcon(toast.type)" 
          :class="[
            'h-5 w-5 mt-0.5 flex-shrink-0',
            toast.type === 'success' && 'text-success-500',
            toast.type === 'error' && 'text-error-500',
            toast.type === 'warning' && 'text-warning-500',
            toast.type === 'info' && 'text-primary-500',
          ]"
        />
        
        <!-- Content -->
        <div class="flex-1">
          <p class="text-sm font-semibold text-text-primary">{{ toast.title }}</p>
          <p v-if="toast.description" class="text-sm text-text-secondary mt-1">
            {{ toast.description }}
          </p>
        </div>
        
        <!-- Close button -->
        <button
          class="flex-shrink-0 p-1 rounded-md text-text-secondary hover:text-text-primary hover:bg-steel-dark transition-colors"
          @click="removeToast(toast.id)"
        >
          <Icon name="lucide:x" class="h-4 w-4" />
        </button>
      </div>
      
      <!-- Progress bar -->
      <div 
        v-if="toast.duration && toast.duration > 0"
        class="absolute bottom-0 left-0 right-0 h-1 rounded-b-lg overflow-hidden"
      >
        <div 
          :class="[
            'h-full animate-toast-progress',
            toast.type === 'success' && 'bg-success-500',
            toast.type === 'error' && 'bg-error-500',
            toast.type === 'warning' && 'bg-warning-500',
            toast.type === 'info' && 'bg-primary-500',
          ]"
          :style="{ animationDuration: `${toast.duration}ms` }"
        />
      </div>
    </div>
  </TransitionGroup>
</template>

<script setup lang="ts">
import { ref } from 'vue'

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
    info: 'lucide:info',
  }
  return icons[type as keyof typeof icons] || 'lucide:info'
}

const addToast = (toast: Omit<Toast, 'id'>) => {
  const id = `toast-${++toastCounter}`
  const newToast: Toast = {
    id,
    duration: 5000,
    ...toast,
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
  removeToast,
})

// Global toast function
if (typeof window !== 'undefined') {
  (window as any).$toast = {
    success: (title: string, description?: string) =>
      addToast({ title, description, type: 'success' }),
    error: (title: string, description?: string) => 
      addToast({ title, description, type: 'error' }),
    warning: (title: string, description?: string) =>
      addToast({ title, description, type: 'warning' }),
    info: (title: string, description?: string) => 
      addToast({ title, description, type: 'info' }),
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

@keyframes toast-progress {
  from {
    width: 100%;
  }
  to {
    width: 0%;
  }
}

.animate-toast-progress {
  animation: toast-progress linear;
}
</style>