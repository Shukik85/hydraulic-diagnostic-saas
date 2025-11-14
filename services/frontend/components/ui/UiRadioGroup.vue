<template>
  <RadioGroupRoot
    v-model="localValue"
    :class="[
      'flex gap-3',
      orientation === 'vertical' ? 'flex-col' : 'flex-row items-center',
      className
    ]"
  >
    <slot />
  </RadioGroupRoot>
</template>

<script setup lang="ts">
import { RadioGroupRoot } from 'radix-vue'
import { ref, watch } from 'vue'

interface Props {
  modelValue?: string | number
  orientation?: 'horizontal' | 'vertical'
  disabled?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  orientation: 'horizontal',
  disabled: false,
})

const emit = defineEmits(['update:modelValue'])

const localValue = ref(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
})

watch(localValue, (newVal) => {
  emit('update:modelValue', newVal)
})
</script>