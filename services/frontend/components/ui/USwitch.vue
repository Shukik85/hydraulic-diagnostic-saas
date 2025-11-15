<template>
  <label
    :class="
      cn(
        'relative inline-flex h-6 w-11 cursor-pointer items-center rounded-full border transition-colors',
        'focus-within:ring-2 focus-within:ring-primary-500/30',
        props.modelValue ? 'bg-gradient-to-r from-primary-600 to-primary-500 border-primary-500' : 'bg-steel-dark border-steel-medium',
        props.disabled && 'cursor-not-allowed opacity-50',
        props.className
      )
    "
  >
    <input
      type="checkbox"
      :checked="props.modelValue"
      :disabled="props.disabled"
      @change="emit('update:modelValue', ($event.target as HTMLInputElement).checked)"
      class="sr-only"
    />
    <span
      :class="
        cn(
          'inline-block h-5 w-5 transform rounded-full transition-all duration-200',
          'shadow-md',
          props.modelValue ? 'translate-x-6 bg-white' : 'translate-x-0.5 bg-steel-shine'
        )
      "
    />
  </label>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  modelValue?: boolean
  disabled?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: false,
  disabled: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()
</script>