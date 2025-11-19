<template>
  <div v-if="steps.length" class="space-y-3">
    <div v-for="(step, idx) in steps" :key="idx" class="p-4 border border-gray-200 rounded-lg bg-white">
      <h4 v-if="step.title" class="font-semibold text-gray-900 mb-2">{{ step.title }}</h4>
      <p class="text-gray-700 whitespace-pre-wrap">{{ step.text }}</p>
      <ul v-if="step.keyPoints.length" class="mt-3 space-y-1 list-disc list-inside text-sm text-gray-600">
        <li v-for="(point, pidx) in step.keyPoints" :key="pidx">{{ point }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from '#imports'
import { validateReasoning } from '~/utils/validation'
import type { ReasoningStep } from '~/types/rag'

interface Props {
  reasoning: string | ReasoningStep[]
}

const props = defineProps<Props>()

interface DisplayStep {
  title: string
  text: string
  keyPoints: string[]
  isConclusion: boolean
}

const steps = computed<DisplayStep[]>(() => {
  const val = validateReasoning(props.reasoning)
  if (typeof val === 'string') {
    // Фолбэк: один step-на-всё
    return [{ title: '', text: val, keyPoints: [], isConclusion: false }]
  }
  // Map ReasoningStep[]
  return val.map(s => ({
    title: s.title,
    text: s.description,
    keyPoints: s.evidence,
    isConclusion: false,
  }))
})
</script>
