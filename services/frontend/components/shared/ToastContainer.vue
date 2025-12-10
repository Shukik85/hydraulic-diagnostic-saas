<script setup lang="ts">
import { computed } from 'vue';
import type { ToastMessage } from '~/types';

const { toasts, remove } = useToast();

const getToastClasses = (type: ToastMessage['type']) => {
  const baseClasses = 'flex items-start gap-3 p-4 rounded-lg shadow-lg border backdrop-blur-sm transition-all';
  
  const variantClasses = {
    success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200',
    error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-200',
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200',
  };
  
  return `${baseClasses} ${variantClasses[type]}`;
};

const getIconName = (type: ToastMessage['type']) => {
  const icons = {
    success: 'heroicons:check-circle',
    error: 'heroicons:x-circle',
    warning: 'heroicons:exclamation-triangle',
    info: 'heroicons:information-circle',
  };
  return icons[type];
};
</script>

<template>
  <div
    class="fixed top-4 right-4 z-50 flex flex-col gap-3 max-w-md w-full pointer-events-none"
    aria-live="polite"
    aria-atomic="true"
  >
    <TransitionGroup name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="getToastClasses(toast.type)"
        class="pointer-events-auto"
        role="alert"
      >
        <!-- Icon -->
        <Icon
          :name="getIconName(toast.type)"
          class="h-5 w-5 flex-shrink-0 mt-0.5"
        />

        <!-- Content -->
        <div class="flex-1 min-w-0">
          <p v-if="toast.title" class="font-semibold text-sm mb-1">
            {{ toast.title }}
          </p>
          <p class="text-sm">
            {{ toast.message }}
          </p>
        </div>

        <!-- Close button -->
        <button
          v-if="toast.dismissible"
          type="button"
          class="flex-shrink-0 rounded-md hover:bg-black/5 dark:hover:bg-white/5 p-1 transition-colors"
          :aria-label="`Dismiss ${toast.type} notification`"
          @click="remove(toast.id)"
        >
          <Icon name="heroicons:x-mark" class="h-4 w-4" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

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
  transform: translateX(100%) scale(0.95);
}

.toast-move {
  transition: transform 0.3s ease;
}
</style>
