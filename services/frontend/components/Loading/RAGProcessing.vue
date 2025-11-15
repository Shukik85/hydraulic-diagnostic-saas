<template>
  <div class="rag-processing">
    <div class="processing-card">
      <!-- Animated Icon -->
      <div class="processing-icon">
        <Icon name="lucide:brain" class="brain-icon" />
        <div class="pulse-ring"></div>
        <div class="pulse-ring pulse-ring-delayed"></div>
      </div>
      
      <h3 class="processing-title">Анализ результатов</h3>
      
      <p class="processing-subtitle">{{ currentStage }}</p>
      
      <!-- Model Badge -->
      <div class="model-badge">
        <Icon name="lucide:sparkles" class="w-4 h-4" />
        <span>DeepSeek-R1</span>
      </div>
      
      <!-- Thinking Steps -->
      <div class="thinking-steps">
        <div 
          v-for="(step, index) in thinkingSteps" 
          :key="index"
          class="thinking-step"
          :class="{ 'step-active': currentStepIndex >= index }"
        >
          <div class="step-dot"></div>
          <span>{{ step }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Props {
  stage?: string
}

const props = withDefaults(defineProps<Props>(), {
  stage: 'Обработка данных...'
})

const thinkingSteps = [
  'Анализ результатов GNN',
  'Изучение контекста оборудования',
  'Формирование интерпретации',
  'Генерация рекомендаций'
]

const currentStepIndex = ref(0)

const currentStage = computed(() => props.stage)

let stepTimer: NodeJS.Timeout | null = null

onMounted(() => {
  // Cycle through steps
  stepTimer = setInterval(() => {
    currentStepIndex.value = (currentStepIndex.value + 1) % thinkingSteps.length
  }, 2000)
})

onUnmounted(() => {
  if (stepTimer) clearInterval(stepTimer)
})
</script>

<style scoped>
.rag-processing {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.processing-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 500px;
  padding: 3rem 2rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  text-align: center;
}

.processing-icon {
  position: relative;
  width: 6rem;
  height: 6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 2rem;
}

.brain-icon {
  width: 3rem;
  height: 3rem;
  color: #6366f1;
  z-index: 2;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

.pulse-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 2px solid #6366f1;
  border-radius: 50%;
  animation: pulse 2s ease-out infinite;
}

.pulse-ring-delayed {
  animation-delay: 1s;
}

@keyframes pulse {
  0% {
    transform: scale(0.5);
    opacity: 1;
  }
  100% {
    transform: scale(1.2);
    opacity: 0;
  }
}

.processing-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #edf2fa;
  margin-bottom: 0.5rem;
}

.processing-subtitle {
  font-size: 1rem;
  color: #bbc6d6;
  margin-bottom: 1.5rem;
}

.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 0.5rem;
  color: #818cf8;
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 2rem;
}

.thinking-steps {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.thinking-step {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: #232b36;
  border: 1px solid #424c5b;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #bbc6d6;
  transition: all 0.3s;
}

.thinking-step.step-active {
  border-color: #6366f1;
  background: rgba(99, 102, 241, 0.05);
  color: #edf2fa;
}

.step-dot {
  width: 0.5rem;
  height: 0.5rem;
  background: #424c5b;
  border-radius: 50%;
  transition: all 0.3s;
}

.step-active .step-dot {
  background: #6366f1;
  box-shadow: 0 0 8px #6366f1;
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>