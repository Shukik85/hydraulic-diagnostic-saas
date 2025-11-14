<script setup lang="ts">
import { ref, computed, watchEffect } from 'vue'
import type { Props } from './RAGInterpretation.vue'
import { validateConfidence, validateSeverity, validateRecommendations, validateReasoning } from '~/utils/validation'

const props = withDefaults(defineProps<Props>(), { loading: false, error: null })

const safeInterpretation = computed(() => {
  if (!props.interpretation) return undefined
  return {
    summary: props.interpretation.summary || '',
    reasoning: validateReasoning(props.interpretation.reasoning),
    recommendations: validateRecommendations(props.interpretation.recommendations),
    severity: validateSeverity(props.interpretation.severity),
    confidence: validateConfidence(props.interpretation.confidence),
    prognosis: props.interpretation.prognosis || null,
    model_version: props.interpretation.model_version,
    processing_time: props.interpretation.processing_time,
    tokens_used: props.interpretation.tokens_used,
    metadata: props.interpretation.metadata || undefined,
  }
})
const recommendations = computed(()=>safeInterpretation.value?.recommendations||[])
const reasoning = computed(()=>safeInterpretation.value?.reasoning||'')
const severity = computed(()=>safeInterpretation.value?.severity||'normal')
const confidence = computed(()=>safeInterpretation.value?.confidence||0)

// ... дальше: остальные вычисляемые свойства, только по safeInterpretation
</script>
