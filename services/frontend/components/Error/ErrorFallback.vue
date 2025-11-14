<template>
  <div class="error-fallback">
    <div class="error-container">
      <div class="error-icon">
        <Icon name="lucide:alert-triangle" class="w-16 h-16" />
      </div>
      
      <h2 class="error-title">Что-то пошло не так</h2>
      
      <p class="error-message">{{ userFriendlyMessage }}</p>
      
      <!-- Technical Details (Collapsible) -->
      <div v-if="error" class="technical-details">
        <button 
          @click="showDetails = !showDetails" 
          class="details-toggle"
        >
          <Icon name="lucide:code" class="w-4 h-4" />
          <span>Технические детали</span>
          <Icon 
            :name="showDetails ? 'lucide:chevron-up' : 'lucide:chevron-down'" 
            class="w-4 h-4 ml-auto"
          />
        </button>
        
        <Transition name="expand">
          <pre v-show="showDetails" class="error-stack">{{ error.stack || error.message }}</pre>
        </Transition>
      </div>
      
      <!-- Actions -->
      <div class="error-actions">
        <button @click="$emit('reset')" class="btn-primary">
          <Icon name="lucide:refresh-cw" class="w-4 h-4" />
          Повторить попытку
        </button>
        <button @click="goHome" class="btn-secondary">
          <Icon name="lucide:home" class="w-4 h-4" />
          На главную
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

interface Props {
  error?: Error | null
}

const props = defineProps<Props>()

defineEmits<{
  reset: []
}>()

const router = useRouter()
const showDetails = ref(false)

const userFriendlyMessage = computed(() => {
  if (!props.error) return 'Неизвестная ошибка'
  
  const message = props.error.message.toLowerCase()
  
  if (message.includes('network') || message.includes('fetch')) {
    return 'Проблема с подключением к серверу. Проверьте интернет-соединение.'
  }
  
  if (message.includes('timeout')) {
    return 'Превышено время ожидания. Попробуйте ещё раз.'
  }
  
  if (message.includes('401') || message.includes('unauthorized')) {
    return 'Сессия истекла. Пожалуйста, войдите заново.'
  }
  
  return 'Произошла ошибка. Попробуйте повторить действие.'
})

const goHome = () => {
  router.push('/')
}
</script>

<style scoped>
.error-fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  padding: 2rem;
}

.error-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 600px;
  padding: 3rem 2rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 1rem;
  text-align: center;
}

.error-icon {
  color: #ef4444;
  margin-bottom: 1.5rem;
}

.error-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #edf2fa;
  margin-bottom: 1rem;
}

.error-message {
  font-size: 1rem;
  line-height: 1.6;
  color: #bbc6d6;
  margin-bottom: 2rem;
}

.technical-details {
  width: 100%;
  margin-bottom: 2rem;
}

.details-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.75rem 1rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
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
  text-align: left;
  overflow-x: auto;
  max-height: 200px;
}

.error-actions {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
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

.btn-primary:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
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

.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  max-height: 0;
}

.expand-enter-to,
.expand-leave-from {
  opacity: 1;
  max-height: 200px;
}
</style>