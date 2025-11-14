<template>
  <div class="reasoning-viewer">
    <!-- Header with Model Badge -->
    <div class="viewer-header">
      <div class="model-badge">
        <Icon name="lucide:sparkles" class="w-5 h-5" />
        <div class="model-info">
          <span class="model-name">DeepSeek-R1</span>
          <span class="model-meta">{{ metadata.version || 'v1.0' }} • {{ metadata.timestamp || 'Just now' }}</span>
        </div>
      </div>
      
      <div class="actions">
        <button @click="copyToClipboard" class="action-btn" title="Копировать">
          <Icon :name="copied ? 'lucide:check' : 'lucide:copy'" class="w-4 h-4" />
        </button>
        <button @click="exportReasoning" class="action-btn" title="Экспорт">
          <Icon name="lucide:download" class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Reasoning Steps -->
    <div class="steps-container">
      <div 
        v-for="(step, index) in steps" 
        :key="index"
        class="step-item"
        :class="{ 'step-conclusion': step.isConclusion }"
      >
        <div class="step-number">
          <Icon 
            :name="step.isConclusion ? 'lucide:lightbulb' : 'lucide:arrow-right'" 
            class="w-4 h-4"
          />
          <span v-if="!step.isConclusion">{{ index + 1 }}</span>
        </div>
        
        <div class="step-content">
          <h4 v-if="step.title" class="step-title">{{ step.title }}</h4>
          <p class="step-text" v-html="highlightKeywords(step.text)"></p>
          
          <!-- Key Points -->
          <div v-if="step.keyPoints && step.keyPoints.length" class="key-points">
            <div 
              v-for="(point, pIndex) in step.keyPoints" 
              :key="pIndex"
              class="key-point"
            >
              <Icon name="lucide:check" class="w-3 h-3" />
              <span>{{ point }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Conclusion Summary (if available) -->
    <div v-if="conclusion" class="conclusion-card">
      <div class="conclusion-header">
        <Icon name="lucide:lightbulb" class="w-6 h-6" />
        <h3>Вывод</h3>
      </div>
      <p class="conclusion-text">{{ conclusion }}</p>
    </div>

    <!-- Raw Reasoning (Collapsible) -->
    <div class="raw-section">
      <button 
        @click="showRaw = !showRaw" 
        class="section-toggle"
        :aria-expanded="showRaw"
      >
        <Icon name="lucide:code" class="w-4 h-4" />
        <span>Полный текст reasoning</span>
        <Icon 
          :name="showRaw ? 'lucide:chevron-up' : 'lucide:chevron-down'" 
          class="w-4 h-4 ml-auto transition-transform"
        />
      </button>
      
      <Transition name="expand">
        <pre v-show="showRaw" class="raw-text">{{ reasoning }}</pre>
      </Transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Step {
  title?: string
  text: string
  keyPoints?: string[]
  isConclusion: boolean
}

interface ModelMetadata {
  version?: string
  timestamp?: string
  tokensUsed?: number
  processingTime?: string
}

interface Props {
  reasoning: string
  metadata?: ModelMetadata
}

const props = withDefaults(defineProps<Props>(), {
  metadata: () => ({})
})

const showRaw = ref(false)
const copied = ref(false)

// Parse reasoning into structured steps
const steps = computed<Step[]>(() => {
  if (!props.reasoning) return []
  
  // Split by common patterns: numbered steps, "Step N:", keywords
  const text = props.reasoning
  const lines = text.split('\n').filter(line => line.trim())
  
  const parsedSteps: Step[] = []
  let currentStep: Step | null = null
  
  lines.forEach((line, index) => {
    const trimmed = line.trim()
    
    // Detect step markers
    const isStepMarker = /^(\d+\.|Step \d+:|\*\*Step \d+\*\*|#+ )/i.test(trimmed)
    const isConclusion = /^(Conclusion:|Summary:|Result:|\*\*Conclusion\*\*)/i.test(trimmed)
    
    if (isStepMarker || isConclusion) {
      // Save previous step
      if (currentStep) {
        parsedSteps.push(currentStep)
      }
      
      // Start new step
      const title = trimmed.replace(/^(\d+\.|Step \d+:|\*\*Step \d+\*\*|#+ )/i, '').trim()
      currentStep = {
        title: title || undefined,
        text: '',
        keyPoints: [],
        isConclusion: isConclusion
      }
    } else if (currentStep) {
      // Add to current step
      if (trimmed.startsWith('-') || trimmed.startsWith('•')) {
        currentStep.keyPoints?.push(trimmed.replace(/^[-•]\s*/, ''))
      } else {
        currentStep.text += (currentStep.text ? ' ' : '') + trimmed
      }
    } else {
      // No step marker yet, start first step
      currentStep = {
        text: trimmed,
        keyPoints: [],
        isConclusion: false
      }
    }
  })
  
  // Add last step
  if (currentStep) {
    parsedSteps.push(currentStep)
  }
  
  return parsedSteps.length > 0 ? parsedSteps : [{
    text: props.reasoning,
    keyPoints: [],
    isConclusion: false
  }]
})

// Extract conclusion if available
const conclusion = computed(() => {
  const conclusionStep = steps.value.find(s => s.isConclusion)
  return conclusionStep?.text || null
})

// Highlight important keywords
const highlightKeywords = (text: string): string => {
  const keywords = [
    'критический',
    'важно',
    'опасно',
    'немедленно',
    'рекомендуется',
    'необходимо',
    'critical',
    'important',
    'urgent',
    'recommended'
  ]
  
  let highlighted = text
  keywords.forEach(keyword => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'gi')
    highlighted = highlighted.replace(regex, '<mark>$&</mark>')
  })
  
  return highlighted
}

// Copy to clipboard
const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(props.reasoning)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  } catch (err) {
    console.error('Failed to copy:', err)
  }
}

// Export reasoning
const exportReasoning = () => {
  const data = {
    model: 'DeepSeek-R1',
    metadata: props.metadata,
    steps: steps.value,
    raw: props.reasoning,
    exportedAt: new Date().toISOString()
  }
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `reasoning-${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.reasoning-viewer {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Header */
.viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-radius: 0.75rem;
}

.model-badge {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: #818cf8;
}

.model-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.model-name {
  font-size: 1rem;
  font-weight: 700;
  color: #edf2fa;
}

.model-meta {
  font-size: 0.75rem;
  color: #bbc6d6;
}

.actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  background: #232b36;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  color: #bbc6d6;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn:hover {
  background: #2b3340;
  border-color: #6366f1;
  color: #818cf8;
}

/* Steps Container */
.steps-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.step-item {
  display: flex;
  gap: 1rem;
  padding: 1.5rem;
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  border: 2px solid #424c5b;
  border-left: 4px solid #6366f1;
  border-radius: 0.75rem;
  transition: all 0.2s;
}

.step-item:hover {
  border-color: #6366f1;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
}

.step-item.step-conclusion {
  border-left-color: #22c55e;
  background: linear-gradient(120deg, rgba(34, 197, 94, 0.05) 0%, #232731 81%);
}

.step-number {
  width: 2.5rem;
  height: 2.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 0.5rem;
  color: #818cf8;
  font-weight: 700;
  flex-shrink: 0;
}

.step-conclusion .step-number {
  background: rgba(34, 197, 94, 0.1);
  border-color: rgba(34, 197, 94, 0.3);
  color: #22c55e;
}

.step-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.step-title {
  font-size: 1rem;
  font-weight: 700;
  color: #818cf8;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.step-conclusion .step-title {
  color: #22c55e;
}

.step-text {
  font-size: 0.875rem;
  line-height: 1.6;
  color: #edf2fa;
}

.step-text :deep(mark) {
  background: rgba(251, 191, 36, 0.2);
  color: #fbbf24;
  padding: 0.125rem 0.25rem;
  border-radius: 0.25rem;
  font-weight: 600;
}

/* Key Points */
.key-points {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.5rem;
  padding-left: 1rem;
  border-left: 2px solid #424c5b;
}

.key-point {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #bbc6d6;
}

.key-point > :first-child {
  color: #22c55e;
  flex-shrink: 0;
}

/* Conclusion Card */
.conclusion-card {
  padding: 1.5rem;
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, #232731 100%);
  border: 2px solid rgba(34, 197, 94, 0.3);
  border-radius: 0.75rem;
  box-shadow: 0 4px 16px rgba(34, 197, 94, 0.1);
}

.conclusion-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  color: #22c55e;
}

.conclusion-header h3 {
  font-size: 1.125rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.conclusion-text {
  font-size: 1rem;
  line-height: 1.6;
  color: #edf2fa;
  font-weight: 500;
}

/* Raw Section */
.raw-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.section-toggle {
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
  width: 100%;
}

.section-toggle:hover {
  background: #2b3340;
  border-color: #6366f1;
}

.raw-text {
  padding: 1.5rem;
  background: #1a1f27;
  border: 1.5px solid #424c5b;
  border-radius: 0.5rem;
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #bbc6d6;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-x: auto;
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
  max-height: 2000px;
}

/* Responsive */
@media (max-width: 768px) {
  .viewer-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .actions {
    width: 100%;
    justify-content: flex-end;
  }
  
  .step-item {
    flex-direction: column;
  }
  
  .step-number {
    width: 100%;
  }
}
</style>