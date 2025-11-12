<!--
  ErrorBoundary.vue — Компонент для graceful error handling
  
  Usage:
  <ErrorBoundary>
    <SomeComponent />
  </ErrorBoundary>
-->
<template>
  <div v-if="error" class="error-boundary">
    <div class="error-content">
      <UIcon name="i-heroicons-exclamation-triangle" class="error-icon" />
      <h3 class="error-title">{{ title }}</h3>
      <p class="error-message">{{ errorMessage }}</p>
      
      <div v-if="showDetails && errorDetails" class="error-details">
        <UAccordion :items="[{
          label: 'Технические детали',
          slot: 'details'
        }]">
          <template #details>
            <pre class="error-stack">{{ errorDetails }}</pre>
          </template>
        </UAccordion>
      </div>
      
      <div class="error-actions">
        <UButton 
          color="primary" 
          @click="reset"
        >
          {{ resetLabel }}
        </UButton>
        
        <UButton 
          v-if="showReload"
          color="gray" 
          variant="outline"
          @click="reload"
        >
          Перезагрузить страницу
        </UButton>
        
        <UButton 
          v-if="onReport"
          color="gray" 
          variant="ghost"
          @click="reportError"
        >
          Сообщить об ошибке
        </UButton>
      </div>
    </div>
  </div>
  <slot v-else />
</template>

<script setup lang="ts">
import { ref, onErrorCaptured, computed } from 'vue'

interface Props {
  title?: string
  resetLabel?: string
  showDetails?: boolean
  showReload?: boolean
  onReport?: (error: Error) => void
  fallback?: any
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Что-то пошло не так',
  resetLabel: 'Попробовать снова',
  showDetails: false,
  showReload: true
})

const error = ref<Error | null>(null)
const errorInfo = ref<string | null>(null)

const errorMessage = computed(() => {
  if (!error.value) return ''
  return error.value.message || 'Произошла неожиданная ошибка'
})

const errorDetails = computed(() => {
  if (!error.value) return ''
  return `${error.value.stack || error.value.toString()}\n\nInfo: ${errorInfo.value || 'N/A'}`
})

onErrorCaptured((err, instance, info) => {
  error.value = err
  errorInfo.value = info
  
  console.error('[ErrorBoundary] Caught error:', err, info)
  
  // Prevent error from propagating
  return false
})

const reset = () => {
  error.value = null
  errorInfo.value = null
}

const reload = () => {
  window.location.reload()
}

const reportError = () => {
  if (props.onReport && error.value) {
    props.onReport(error.value)
  }
}
</script>

<style scoped>
.error-boundary {
  @apply flex items-center justify-center min-h-[400px] p-8;
}

.error-content {
  @apply max-w-lg w-full text-center;
}

.error-icon {
  @apply w-16 h-16 text-red-500 mx-auto mb-4;
}

.error-title {
  @apply text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2;
}

.error-message {
  @apply text-gray-600 dark:text-gray-400 mb-6;
}

.error-details {
  @apply mb-6 text-left;
}

.error-stack {
  @apply text-xs bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto max-h-[200px];
}

.error-actions {
  @apply flex gap-3 justify-center flex-wrap;
}
</style>
