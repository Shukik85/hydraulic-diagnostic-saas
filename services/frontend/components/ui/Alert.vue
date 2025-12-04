<script setup lang="ts">
import { ref } from 'vue';

interface AlertProps {
  variant?: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  dismissible?: boolean;
  icon?: string;
}

const props = withDefaults(defineProps<AlertProps>(), {
  variant: 'info',
  dismissible: false,
});

const emit = defineEmits<{
  dismiss: [];
}>();

const isVisible = ref(true);

const handleDismiss = () => {
  isVisible.value = false;
  emit('dismiss');
};

const variantClasses = {
  info: {
    container: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    icon: 'text-blue-600 dark:text-blue-400',
    title: 'text-blue-900 dark:text-blue-100',
    description: 'text-blue-800 dark:text-blue-200',
  },
  success: {
    container: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    icon: 'text-green-600 dark:text-green-400',
    title: 'text-green-900 dark:text-green-100',
    description: 'text-green-800 dark:text-green-200',
  },
  warning: {
    container: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
    icon: 'text-amber-600 dark:text-amber-400',
    title: 'text-amber-900 dark:text-amber-100',
    description: 'text-amber-800 dark:text-amber-200',
  },
  error: {
    container: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    icon: 'text-red-600 dark:text-red-400',
    title: 'text-red-900 dark:text-red-100',
    description: 'text-red-800 dark:text-red-200',
  },
};

const defaultIcons = {
  info: 'heroicons:information-circle',
  success: 'heroicons:check-circle',
  warning: 'heroicons:exclamation-triangle',
  error: 'heroicons:x-circle',
};

const iconName = props.icon || defaultIcons[props.variant];
</script>

<template>
  <Transition
    enter-active-class="transition ease-out duration-200"
    enter-from-class="opacity-0 scale-95"
    enter-to-class="opacity-100 scale-100"
    leave-active-class="transition ease-in duration-150"
    leave-from-class="opacity-100 scale-100"
    leave-to-class="opacity-0 scale-95"
  >
    <div
      v-if="isVisible"
      :class="[
        'rounded-lg border p-4',
        variantClasses[variant].container,
      ]"
      :role="variant === 'error' ? 'alert' : 'status'"
      :aria-live="variant === 'error' ? 'assertive' : 'polite'"
    >
      <div class="flex gap-3">
        <!-- Icon -->
        <Icon
          :name="iconName"
          class="h-5 w-5 shrink-0"
          :class="variantClasses[variant].icon"
          aria-hidden="true"
        />

        <!-- Content -->
        <div class="flex-1">
          <h4
            v-if="title || $slots.title"
            class="text-sm font-semibold mb-1"
            :class="variantClasses[variant].title"
          >
            <slot name="title">
              {{ title }}
            </slot>
          </h4>

          <div
            class="text-sm"
            :class="variantClasses[variant].description"
          >
            <slot />
          </div>
        </div>

        <!-- Dismiss Button -->
        <button
          v-if="dismissible"
          type="button"
          class="shrink-0 opacity-70 hover:opacity-100 transition-opacity"
          :class="variantClasses[variant].icon"
          aria-label="Dismiss alert"
          @click="handleDismiss"
        >
          <Icon name="heroicons:x-mark" class="h-5 w-5" aria-hidden="true" />
        </button>
      </div>
    </div>
  </Transition>
</template>
