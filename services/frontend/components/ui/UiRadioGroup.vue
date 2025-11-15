<script setup lang="ts">
import { RadioGroupRoot } from 'radix-vue'

interface Props {
  modelValue?: string | number
  orientation?: 'horizontal' | 'vertical'
  disabled?: boolean
  className?: string
}

interface Emits {
  (e: 'update:modelValue', value: string | number): void
}

const props = withDefaults(defineProps<Props>(), {
  orientation: 'horizontal',
  disabled: false
})

const emit = defineEmits<Emits>()

const localValue = ref<string | number | undefined>(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
})

watch(localValue, (newVal) => {
  if (newVal !== undefined) {
    emit('update:modelValue', newVal)
  }
})
</script>

<template>
  <RadioGroupRoot
    v-model="localValue"
    :class="[
      'flex gap-3',
      props.orientation === 'vertical' ? 'flex-col' : 'flex-row items-center',
      props.className
    ]"
  >
    <slot />
  </RadioGroupRoot>
</template>
