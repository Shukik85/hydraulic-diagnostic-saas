<template>
  <div class="network-error">
    <div class="error-container">
      <div class="error-animation">
        <Icon name="lucide:wifi-off" class="w-20 h-20" />
      </div>
      
      <h2 class="error-title">Нет подключения</h2>
      
      <p class="error-message">{{ message }}</p>
      
      <!-- Connection Status -->
      <div class="status-indicator">
        <div class="status-dot" :class="statusClass"></div>
        <span class="status-text">{{ statusText }}</span>
      </div>
      
      <!-- Troubleshooting Tips -->
      <div class="tips-section">
        <h4 class="tips-title">Что можно проверить:</h4>
        <ul class="tips-list">
          <li>
            <Icon name="lucide:check" class="w-4 h-4" />
            <span>Подключение к интернету</span>
          </li>
          <li>
            <Icon name="lucide:check" class="w-4 h-4" />
            <span>Доступность сервера</span>
          </li>
          <li>
            <Icon name="lucide:check" class="w-4 h-4" />
            <span>Настройки файервола</span>
          </li>
          <li>
            <Icon name="lucide:check" class="w-4 h-4" />
            <span>VPN или proxy настройки</span>
          </li>
        </ul>
      </div>
      
      <!-- Actions -->
      <div class="error-actions">
        <button @click="retry" class="btn-primary" :disabled="isRetrying">
          <Icon name="lucide:refresh-cw" class="w-4 h-4" :class="{ 'animate-spin': isRetrying }" />
          {{ isRetrying ? 'Проверка...' : 'Проверить снова' }}
        </button>
        <button v-if="showOfflineMode" @click="$emit('offline-mode')" class="btn-secondary">
          <Icon name="lucide:download-cloud" class="w-4 h-4" />
          Режим offline
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Props {
  message?: string
  showOfflineMode?: boolean
  autoRetry?: boolean
  retryInterval?: number
}

const props = withDefaults(defineProps<Props>(), {
  message: 'Не удаётся связаться с сервером. Проверьте подключение.',
  showOfflineMode: false,
  autoRetry: true,
  retryInterval: 30
})

defineEmits<{
  retry: []
  'offline-mode': []
}>()

const isRetrying = ref(false)
const retryIn = ref(props.retryInterval)
const isOnline = ref(navigator.onLine)

const statusClass = computed(() => {
  return isOnline.value ? 'status-checking' : 'status-offline'
})

const statusText = computed(() => {
  return isOnline.value ? 'Проверка соединения...' : 'Нет интернета'
})

const retry = async () => {
  isRetrying.value = true
  // Emit retry event
  await new Promise(resolve => setTimeout(resolve, 1000))
  isRetrying.value = false
}

const handleOnline = () => { isOnline.value = true }
const handleOffline = () => { isOnline.value = false }

onMounted(() => {
  window.addEventListener('online', handleOnline)
  window.addEventListener('offline', handleOffline)
})

onUnmounted(() => {
  window.removeEventListener('online', handleOnline)
  window.removeEventListener('offline', handleOffline)
})
</script>

<style scoped>
.network-error {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 500px;
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

.error-animation {
  color: #ef4444;
  margin-bottom: 2rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
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
  margin-bottom: 1.5rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1.5rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  margin-bottom: 2rem;
}

.status-dot {
  width: 0.75rem;
  height: 0.75rem;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-dot.status-offline {
  background: #ef4444;
  box-shadow: 0 0 8px #ef4444;
}

.status-dot.status-checking {
  background: #fbbf24;
  box-shadow: 0 0 8px #fbbf24;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.status-text {
  font-size: 0.875rem;
  font-weight: 600;
  color: #edf2fa;
}

.tips-section {
  width: 100%;
  text-align: left;
  margin-bottom: 2rem;
}

.tips-title {
  font-size: 0.875rem;
  font-weight: 700;
  color: #818cf8;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.75rem;
}

.tips-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  list-style: none;
}

.tips-list li {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.875rem;
  color: #bbc6d6;
}

.tips-list li > :first-child {
  color: #6366f1;
  flex-shrink: 0;
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
  max-height: 150px;
}
</style>