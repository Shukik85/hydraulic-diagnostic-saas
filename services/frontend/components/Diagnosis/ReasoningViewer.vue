<script setup lang="ts">
import { computed } from 'vue'
import { validateReasoning } from '~/utils/validation'
import type { Props } from './ReasoningViewer.vue'

const props = defineProps<Props>()
const steps = computed(() => {
  const val = validateReasoning(props.reasoning)
  if (typeof val === 'string') {
    // Фолбэк: один step-на-всё
    return [ { title: '', text: val, keyPoints: [], isConclusion: false } ]
  }
  // Map ReasoningStep[]
  return val.map(s => ({
    title: s.title,
    text: s.description,
    keyPoints: s.evidence,
    isConclusion: false
  }))
})
</script>