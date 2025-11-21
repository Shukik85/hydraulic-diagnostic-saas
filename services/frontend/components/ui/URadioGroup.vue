<script setup lang="ts">
import { ref, computed } from '#imports'
import { RadioGroupRoot } from 'radix-vue'

interface Props {
  modelValue: string | number
  name?: string
  required?: boolean
  disabled?: boolean
  orientation?: 'horizontal' | 'vertical'
}

const props = withDefaults(defineProps<Props>(), {
  orientation: 'vertical',
  required: false,
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

// Конвертируем number в string для radix-vue
const localValue = computed({
  get: () => String(props.modelValue),
  set: (val: string) => emit('update:modelValue', val)
})
</script>

<template>
  <RadioGroupRoot v-model="localValue" :class="[
    'flex gap-2',
    orientation === 'horizontal' ? 'flex-row' : 'flex-col'
  ]" :name="name" :required="required" :disabled="disabled">
    <slot />
  </RadioGroupRoot>
</template>
