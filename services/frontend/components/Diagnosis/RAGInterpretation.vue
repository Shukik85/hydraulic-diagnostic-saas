<template>
  <div class="rag-interpretation">
    <!-- Loading State -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p class="loading-text">Анализ результатов диагностики...</p>
      <p class="loading-subtext">DeepSeek-R1 обрабатывает данные</p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="error-container">
      <div class="error-icon">⚠</div>
      <h3 class="error-title">Интерпретация недоступна</h3>
      <p class="error-message">{{ error }}</p>
      <button @click="$emit('retry')" class="retry-btn">
        <Icon name="lucide:refresh-cw" class="w-4 h-4" />
        Повторить попытку
      </button>
    </div>

    <!-- Empty State -->
    <div v-else-if="!interpretation" class="empty-container">
      <div class="empty-icon">🤖</div>
      <p class="empty-text">Нет данных для интерпретации</p>
    </div>

    <!-- Main Content -->
    <div v-else class="interpretation-content">
      <!-- Summary Card -->
      <div class="summary-card">
        <div class="summary-header">
          <div class="health-indicator" :class="severityClass">
            <Icon :name="severityIcon" class="w-6 h-6" />
          </div>
          <div class="summary-meta">
            <h3 class="summary-title">Интерпретация диагностики</h3>
            <div class="badges">
              <span class="badge" :class="severityClass">{{ severityLabel }}</span>
              <span class="badge badge-confidence">{{ confidencePercent }}% уверенность</span>
            </div>
          </div>
        </div>
        <p class="summary-text">{{ interpretation.summary }}</p>
      </div>

      <!-- Reasoning Section (Expandable) -->
      <div class="reasoning-section">
        <button 
          @click="showReasoning = !showReasoning" 
          class="section-toggle"
          :aria-expanded="showReasoning"
        >
          <Icon name="lucide:brain" class="w-5 h-5" />
          <span>Процесс анализа</span>
          <Icon 
            :name="showReasoning ? 'lucide:chevron-up' : 'lucide:chevron-down'" 
            class="w-5 h-5 ml-auto transition-transform"
          />
        </button>
        
        <Transition name="expand">
          <div v-show="showReasoning" class="reasoning-content">
            <div class="model-badge">
              <Icon name="lucide:sparkles" class="w-4 h-4" />
              <span>DeepSeek-R1</span>
            </div>
            <pre class="reasoning-text">{{ interpretation.reasoning }}</pre>
          </div>
        </Transition>
      </div>

      <!-- Recommendations -->
      <div class="recommendations-section">
        <h4 class="section-title">
          <Icon name="lucide:list-checks" class="w-5 h-5" />
          Рекомендации ({{ recommendations.length }})
        </h4>
        <ul class="recommendations-list">
          <li 
            v-for="(rec, index) in recommendations" 
            :key="index"
            class="recommendation-item"
            :class="getRecommendationPriority(index)"
          >
            <div class="rec-icon">
              <Icon :name="getRecommendationIcon(index)" class="w-5 h-5" />
            </div>
            <div class="rec-content">
              <span class="rec-priority">{{ getPriorityLabel(index) }}</span>
              <p class="rec-text">{{ rec }}</p>
            </div>
          </li>
        </ul>
      </div>

      <!-- Prognosis Card -->
      <div v-if="interpretation.prognosis" class="prognosis-card">
        <h4 class="section-title">
          <Icon name="lucide:trending-up" class="w-5 h-5" />
          Прогноз
        </h4>
        <p class="prognosis-text">{{ interpretation.prognosis }}</p>
        
        <!-- Confidence Meter -->
        <div class="confidence-meter">
          <div class="confidence-label">
            <span>Уверенность в прогнозе</span>
            <span class="confidence-value">{{ confidencePercent }}%</span>
          </div>
          <div class="confidence-bar">
            <div 
              class="confidence-fill" 
              :style="{ width: confidencePercent + '%' }"
              :class="confidenceClass"
            ></div>
          </div>
        </div>
      </div>

      <!-- Technical Details (Collapsible) -->
      <div class="technical-section">
        <button 
          @click="showTechnical = !showTechnical" 
          class="section-toggle"
          :aria-expanded="showTechnical"
        >
          <Icon name="lucide:settings" class="w-5 h-5" />
          <span>Технические детали</span>
          <Icon 
            :name="showTechnical ? 'lucide:chevron-up' : 'lucide:chevron-down'" 
            class="w-5 h-5 ml-auto transition-transform"
          />
        </button>
        
        <Transition name="expand">
          <div v-show="showTechnical" class="technical-content">
            <div class="technical-grid">
              <div class="tech-item">
                <span class="tech-label">Модель:</span>
                <span class="tech-value">DeepSeek-R1</span>
              </div>
              <div class="tech-item">
                <span class="tech-label">Версия:</span>
                <span class="tech-value">{{ interpretation.model_version || '1.0.0' }}</span>
              </div>
              <div class="tech-item">
                <span class="tech-label">Время анализа:</span>
                <span class="tech-value">{{ interpretation.processing_time || 'N/A' }}</span>
              </div>
              <div class="tech-item">
                <span class="tech-label">Токены:</span>
                <span class="tech-value">{{ interpretation.tokens_used || 'N/A' }}</span>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface RAGInterpretation {
  summary: string
  reasoning: string
  recommendations: string[]
  prognosis?: string
  severity: 'normal' | 'warning' | 'critical'
  confidence: number
  model_version?: string
  processing_time?: string
  tokens_used?: number
}

interface Props {
  interpretation?: RAGInterpretation
  loading?: boolean
  error?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  error: null
})

defineEmits<{
  retry: []
}>()

const showReasoning = ref(false)
const showTechnical = ref(false)

const recommendations = computed(() => props.interpretation?.recommendations || [])

const confidencePercent = computed(() => {
  return Math.round((props.interpretation?.confidence || 0) * 100)
})

const severityClass = computed(() => {
  const severity = props.interpretation?.severity || 'normal'
  return `severity-${severity}`
})

const confidenceClass = computed(() => {
  const conf = confidencePercent.value
  if (conf >= 80) return 'confidence-high'
  if (conf >= 60) return 'confidence-medium'
  return 'confidence-low'
})

const severityIcon = computed(() => {
  const severity = props.interpretation?.severity || 'normal'
  const icons = {
    normal: 'lucide:check-circle',
    warning: 'lucide:alert-triangle',
    critical: 'lucide:alert-octagon'
  }
  return icons[severity]
})

const severityLabel = computed(() => {
  const severity = props.interpretation?.severity || 'normal'
  const labels = {
    normal: 'Норма',
    warning: 'Требует внимания',
    critical: 'Критично'
  }
  return labels[severity]
})

const getRecommendationPriority = (index: number) => {
  if (index === 0) return 'priority-high'
  if (index < 3) return 'priority-medium'
  return 'priority-low'
}

const getRecommendationIcon = (index: number) => {
  if (index === 0) return 'lucide:alert-circle'
  if (index < 3) return 'lucide:wrench'
  return 'lucide:info'
}

const getPriorityLabel = (index: number) => {
  if (index === 0) return 'Срочно'
  if (index < 3) return 'Важно'
  return 'Рекомендуется'
}
</script>

<style scoped>
.rag-interpretation {
  width: 100%;
}

/* Loading State */
.loading-container,
.error-container,
.empty-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem 2rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  text-align: center;
}

.loading-spinner {
  width: 3rem;
  height: 3rem;
  border: 3px solid #424c5b;
  border-top-color: #6366f1;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-text {
  margin-top: 1rem;
  font-size: 1rem;
  font-weight: 600;
  color: #edf2fa;
}

.loading-subtext {
  margin-top: 0.5rem;
  font-size: 0.875rem;
  color: #bbc6d6;
}

/* Error State */
.error-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.error-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ef4444;
  margin-bottom: 0.5rem;
}

.error-message {
  color: #bbc6d6;
  margin-bottom: 1.5rem;
}

.retry-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1.25rem;
  background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.retry-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
}

/* Empty State */
.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.empty-text {
  font-size: 1rem;
  color: #bbc6d6;
}

/* Main Content */
.interpretation-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Summary Card */
.summary-card {
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 2px 12px rgba(61, 72, 102, 0.18);
}

.summary-header {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
}

.health-indicator {
  width: 3rem;
  height: 3rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.5rem;
  flex-shrink: 0;
}

.health-indicator.severity-normal {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.4);
}

.health-indicator.severity-warning {
  background: rgba(251, 191, 36, 0.1);
  color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.4);
}

.health-indicator.severity-critical {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.4);
}

.summary-meta {
  flex: 1;
}

.summary-title {
  font-size: 1.125rem;
  font-weight: 700;
  color: #edf2fa;
  margin-bottom: 0.5rem;
}

.badges {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.badge.severity-normal {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.4);
}

.badge.severity-warning {
  background: rgba(251, 191, 36, 0.1);
  color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.4);
}

.badge.severity-critical {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.4);
}

.badge-confidence {
  background: rgba(99, 102, 241, 0.1);
  color: #818cf8;
  border: 1px solid rgba(99, 102, 241, 0.4);
}

.summary-text {
  font-size: 1rem;
  line-height: 1.6;
  color: #edf2fa;
}

/* Section Toggle */
.section-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  color: #edf2fa;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.section-toggle:hover {
  background: #2b3340;
  border-color: #6366f1;
}

/* Reasoning Section */
.reasoning-section,
.technical-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.reasoning-content,
.technical-content {
  padding: 1.5rem;
  background: #1a1f27;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
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
  margin-bottom: 1rem;
}

.reasoning-text {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #bbc6d6;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* Recommendations */
.recommendations-section {
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  padding: 1.5rem;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1rem;
  font-weight: 700;
  color: #edf2fa;
  margin-bottom: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  list-style: none;
}

.recommendation-item {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  transition: all 0.2s;
}

.recommendation-item:hover {
  border-color: #6366f1;
  transform: translateX(4px);
}

.recommendation-item.priority-high {
  border-left: 4px solid #ef4444;
}

.recommendation-item.priority-medium {
  border-left: 4px solid #fbbf24;
}

.recommendation-item.priority-low {
  border-left: 4px solid #6366f1;
}

.rec-icon {
  width: 2.5rem;
  height: 2.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 0.5rem;
  color: #818cf8;
  flex-shrink: 0;
}

.rec-content {
  flex: 1;
}

.rec-priority {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #818cf8;
  margin-bottom: 0.25rem;
}

.rec-text {
  font-size: 0.875rem;
  line-height: 1.5;
  color: #edf2fa;
}

/* Prognosis Card */
.prognosis-card {
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
  padding: 1.5rem;
}

.prognosis-text {
  font-size: 0.875rem;
  line-height: 1.6;
  color: #edf2fa;
  margin-bottom: 1.5rem;
}

.confidence-meter {
  margin-top: 1rem;
}

.confidence-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.875rem;
  color: #bbc6d6;
  margin-bottom: 0.5rem;
}

.confidence-value {
  font-weight: 700;
  color: #edf2fa;
}

.confidence-bar {
  height: 0.5rem;
  background: #232b36;
  border-radius: 9999px;
  overflow: hidden;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.confidence-fill {
  height: 100%;
  transition: width 0.5s ease;
  box-shadow: 0 0 10px currentColor;
}

.confidence-fill.confidence-high {
  background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
}

.confidence-fill.confidence-medium {
  background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
}

.confidence-fill.confidence-low {
  background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
}

/* Technical Details */
.technical-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.tech-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.tech-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #bbc6d6;
}

.tech-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: #edf2fa;
}

/* Transitions */
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
  max-height: 1000px;
}

/* Responsive */
@media (max-width: 768px) {
  .summary-header {
    flex-direction: column;
  }
  
  .technical-grid {
    grid-template-columns: 1fr;
  }
  
  .recommendation-item {
    flex-direction: column;
  }
}
</style>