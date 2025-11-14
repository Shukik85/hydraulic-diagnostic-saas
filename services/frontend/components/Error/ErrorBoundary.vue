<template>
  <div>
    <slot v-if="!hasError" />
    <ErrorFallback 
      v-else
      :error="error"
      @reset="reset"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'
import ErrorFallback from './ErrorFallback.vue'

const hasError = ref(false)
const error = ref<Error | null>(null)

onErrorCaptured((err: Error) => {
  hasError.value = true
  error.value = err
  
  // Log to error reporting service
  console.error('[ErrorBoundary]', err)
  
  // Prevent propagation
  return false
})

const reset = () => {
  hasError.value = false
  error.value = null
}
</script>