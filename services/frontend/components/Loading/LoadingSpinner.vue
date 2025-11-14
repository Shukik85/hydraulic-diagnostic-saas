<template>
  <div class="loading-spinner" :class="[sizeClass, variantClass]">
    <div class="spinner">
      <div class="spinner-ring"></div>
      <Icon v-if="icon" :name="icon" class="spinner-icon" />
    </div>
    <p v-if="text" class="spinner-text">{{ text }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'primary' | 'success' | 'warning' | 'error'
  text?: string
  icon?: string
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  variant: 'primary'
})

const sizeClass = computed(() => `spinner-${props.size}`)
const variantClass = computed(() => `spinner-${props.variant}`)
</script>

<style scoped>
.loading-spinner {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}

.spinner {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner-ring {
  border: 3px solid #424c5b;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.spinner-sm .spinner-ring {
  width: 1.5rem;
  height: 1.5rem;
  border-width: 2px;
}

.spinner-md .spinner-ring {
  width: 3rem;
  height: 3rem;
  border-width: 3px;
}

.spinner-lg .spinner-ring {
  width: 4rem;
  height: 4rem;
  border-width: 4px;
}

.spinner-xl .spinner-ring {
  width: 6rem;
  height: 6rem;
  border-width: 5px;
}

.spinner-primary .spinner-ring {
  border-top-color: #6366f1;
}

.spinner-success .spinner-ring {
  border-top-color: #22c55e;
}

.spinner-warning .spinner-ring {
  border-top-color: #fbbf24;
}

.spinner-error .spinner-ring {
  border-top-color: #ef4444;
}

.spinner-icon {
  position: absolute;
  color: #6366f1;
}

.spinner-sm .spinner-icon { width: 0.75rem; height: 0.75rem; }
.spinner-md .spinner-icon { width: 1.5rem; height: 1.5rem; }
.spinner-lg .spinner-icon { width: 2rem; height: 2rem; }
.spinner-xl .spinner-icon { width: 3rem; height: 3rem; }

.spinner-text {
  font-size: 0.875rem;
  font-weight: 600;
  color: #bbc6d6;
  text-align: center;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>