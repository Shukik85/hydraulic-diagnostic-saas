<template>
  <div class="rag-error-state">
    <div class="error-card">
      <div class="error-header">
        <div class="error-icon-circle">
          <Icon name="lucide:bot-off" class="w-8 h-8" />
        </div>
        <div class="error-content">
          <h3 class="error-title">Интерпретация RAG недоступна</h3>
          <p class="error-subtitle">{{ errorMessage }}</p>
        </div>
      </div>

      <!-- Error Details -->
      <div class="error-details">
        <div class="detail-item">
          <Icon name="lucide:info" class="w-4 h-4" />
          <span>Результаты GNN доступны и могут использоваться</span>
        </div>
        <div class="detail-item">
          <Icon name="lucide:clock" class="w-4 h-4" />
          <span>Автоматическая повторная попытка через {{ retryIn }}s</span>
        </div>
      </div>

      <!-- Fallback Mode Info -->
      <div class="fallback-info">
        <Icon name="lucide:shield-check" class="w-5 h-5" />
        <div class="fallback-text">
          <strong>Режим fallback активен:</strong>
          <p>Используются только числовые результаты от GNN модели. Интерпретация будет добавлена позже.</p>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="error-actions">
        <button @click="$emit('retry')" class="btn-primary" :disabled="isRetrying">
          <Icon name="lucide:refresh-cw" class="w-4 h-4" :class="{ 'animate-spin': isRetrying }" />
          {{ isRetrying ? 'Повтор...' : 'Повторить сейчас' }}
        </button>
        <button @click="$emit('continue')" class="btn-secondary">
          <Icon name="lucide:arrow-right" class="w-4 h-4" />
          Продолжить без RAG
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Props {
  error?: Error | string | null
  autoRetry?: boolean
  retryDelay?: number
}

const props = withDefaults(defineProps<Props>(), {
  autoRetry: true,
  retryDelay: 30
})

defineEmits<{
  retry: []
  continue: []
}>()

const showDetails = ref(false)
const isRetrying = ref(false)
const retryIn = ref(props.retryDelay)
let retryTimer: NodeJS.Timeout | null = null

const errorMessage = computed(() => {
  if (typeof props.error === 'string') return props.error
  if (props.error instanceof Error) return props.error.message
  return 'Сервис RAG временно недоступен'
})

onMounted(() => {
  if (props.autoRetry) {
    retryTimer = setInterval(() => {
      retryIn.value--
      if (retryIn.value <= 0) {
        clearInterval(retryTimer!)
        // Auto-emit retry
      }
    }, 1000)
  }
})

onUnmounted(() => {
  if (retryTimer) clearInterval(retryTimer)
})
</script>

<style scoped>
.rag-error-state {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.error-card {
  width: 100%;
  max-width: 700px;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  padding: 2rem;
  box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
}

.error-header {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.error-icon-circle {
  width: 4rem;
  height: 4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(239, 68, 68, 0.1);
  border: 2px solid rgba(239, 68, 68, 0.3);
  border-radius: 0.75rem;
  color: #ef4444;
  flex-shrink: 0;
}

.error-content {
  flex: 1;
}

.error-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #edf2fa;
  margin-bottom: 0.5rem;
}

.error-subtitle {
  font-size: 0.875rem;
  color: #bbc6d6;
  line-height: 1.5;
}

.error-details {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
}

.detail-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.875rem;
  color: #bbc6d6;
}

.detail-item > :first-child {
  color: #6366f1;
  flex-shrink: 0;
}

.fallback-info {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 0.5rem;
  margin-bottom: 1.5rem;
}

.fallback-info > :first-child {
  color: #818cf8;
  flex-shrink: 0;
  margin-top: 0.25rem;
}

.fallback-text {
  flex: 1;
  font-size: 0.875rem;
}

.fallback-text strong {
  color: #edf2fa;
  display: block;
  margin-bottom: 0.25rem;
}

.fallback-text p {
  color: #bbc6d6;
  line-height: 1.5;
}

.technical-details {
  margin-bottom: 1.5rem;
}

.details-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.75rem 1rem;
  background: #1a1f27;
  border: 1px solid #424c5b;
  border-radius: 0.5rem;
  color: #bbc6d6;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
}

.details-toggle:hover {
  border-color: #6366f1;
}

.error-stack {
  margin-top: 0.75rem;
  padding: 1rem;
  background: #1a1f27;
  border: 1px solid #424c5b;
  border-radius: 0.5rem;
  font-family: monospace;
  font-size: 0.75rem;
  color: #ef4444;
  overflow-x: auto;
  max-height: 150px;
}

.error-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.btn-primary,
.btn-secondary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
  color: white;
  border: none;
}

.btn-primary:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background: #232b36;
  color: #edf2fa;
  border: 1.5px solid #424c5b;
}

.btn-secondary:hover {
  background: #2b3340;
  border-color: #6366f1;
}

.animate-spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  max-height: 0;
}

.expand-enter-to,
.expand-leave-from {
  opacity: 1;
  max-height: 150px;
}

@media (max-width: 768px) {
  .error-header {
    flex-direction: column;
    text-align: center;
  }
  
  .error-actions {
    flex-direction: column;
  }
  
  .btn-primary,
  .btn-secondary {
    width: 100%;
  }
}
</style>