<template>
  <div class="diagnosis-progress">
    <div class="progress-card">
      <!-- Header -->
      <div class="progress-header">
        <h3 class="progress-title">Диагностика в процессе</h3>
        <span class="progress-percentage">{{ totalProgress }}%</span>
      </div>

      <!-- Overall Progress Bar -->
      <div class="progress-bar-container">
        <div class="progress-bar">
          <div 
            class="progress-fill" 
            :style="{ width: totalProgress + '%' }"
          ></div>
        </div>
        <div v-if="eta" class="progress-eta">
          <Icon name="lucide:clock" class="w-3 h-3" />
          <span>Осталось ~{{ eta }}</span>
        </div>
      </div>

      <!-- Stages -->
      <div class="stages-container">
        <div 
          v-for="(stage, index) in stages" 
          :key="stage.id"
          class="stage-item"
          :class="getStageClass(stage)"
        >
          <div class="stage-indicator">
            <Icon 
              v-if="stage.status === 'complete'" 
              name="lucide:check" 
              class="w-4 h-4"
            />
            <Icon 
              v-else-if="stage.status === 'error'" 
              name="lucide:x" 
              class="w-4 h-4"
            />
            <div v-else-if="stage.status === 'active'" class="spinner-small"></div>
            <span v-else class="stage-number">{{ index + 1 }}</span>
          </div>
          
          <div class="stage-content">
            <div class="stage-name">{{ stage.name }}</div>
            <div v-if="stage.status === 'active' && stage.progress" class="stage-progress">
              {{ stage.progress }}%
            </div>
          </div>
          
          <div v-if="stage.duration" class="stage-duration">
            {{ stage.duration }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Stage {
  id: string
  name: string
  status: 'pending' | 'active' | 'complete' | 'error'
  progress?: number
  duration?: string
}

interface Props {
  stages: Stage[]
  eta?: string
}

const props = defineProps<Props>()

const totalProgress = computed(() => {
  const completedStages = props.stages.filter(s => s.status === 'complete').length
  const activeStage = props.stages.find(s => s.status === 'active')
  
  let progress = (completedStages / props.stages.length) * 100
  
  if (activeStage && activeStage.progress) {
    progress += (activeStage.progress / 100) * (100 / props.stages.length)
  }
  
  return Math.round(progress)
})

const getStageClass = (stage: Stage) => {
  return `stage-${stage.status}`
}
</script>

<style scoped>
.diagnosis-progress {
  width: 100%;
}

.progress-card {
  padding: 2rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  box-shadow: 0 2px 12px rgba(61, 72, 102, 0.18);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.progress-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #edf2fa;
}

.progress-percentage {
  font-size: 1.5rem;
  font-weight: 800;
  color: #6366f1;
}

.progress-bar-container {
  margin-bottom: 2rem;
}

.progress-bar {
  height: 0.75rem;
  background: #232b36;
  border-radius: 9999px;
  overflow: hidden;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
  box-shadow: 0 0 12px rgba(99, 102, 241, 0.6);
  transition: width 0.5s ease;
}

.progress-eta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: #bbc6d6;
}

.stages-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.stage-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  transition: all 0.2s;
}

.stage-item.stage-active {
  border-color: #6366f1;
  background: rgba(99, 102, 241, 0.05);
}

.stage-item.stage-complete {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.05);
}

.stage-item.stage-error {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.05);
}

.stage-indicator {
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.5rem;
  flex-shrink: 0;
}

.stage-pending .stage-indicator {
  background: #232b36;
  border: 1.5px solid #424c5b;
}

.stage-active .stage-indicator {
  background: rgba(99, 102, 241, 0.1);
  border: 1.5px solid rgba(99, 102, 241, 0.3);
  color: #818cf8;
}

.stage-complete .stage-indicator {
  background: rgba(34, 197, 94, 0.1);
  border: 1.5px solid rgba(34, 197, 94, 0.3);
  color: #22c55e;
}

.stage-error .stage-indicator {
  background: rgba(239, 68, 68, 0.1);
  border: 1.5px solid rgba(239, 68, 68, 0.3);
  color: #ef4444;
}

.stage-number {
  font-size: 0.875rem;
  font-weight: 700;
  color: #bbc6d6;
}

.spinner-small {
  width: 1rem;
  height: 1rem;
  border: 2px solid #424c5b;
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.stage-content {
  flex: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stage-name {
  font-size: 0.875rem;
  font-weight: 600;
  color: #edf2fa;
}

.stage-progress {
  font-size: 0.75rem;
  color: #818cf8;
  font-weight: 700;
}

.stage-duration {
  font-size: 0.75rem;
  color: #bbc6d6;
}

@media (max-width: 768px) {
  .progress-card {
    padding: 1.5rem;
  }
  
  .stage-content {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.25rem;
  }
}
</style>