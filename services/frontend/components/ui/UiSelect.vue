<template>
  <div class="relative w-full">
    <select
      v-model="localValue"
      @change="handleChange"
      :disabled="disabled"
      :class="[
        'input-metal appearance-none w-full pr-10',
        'cursor-pointer',
        disabled && 'cursor-not-allowed opacity-50',
        className
      ]"
      v-bind="$attrs"
    >
      <slot />
    </select>
    <div class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
      <svg 
        class="w-5 h-5 text-steel-light" 
        fill="none" 
        stroke="currentColor" 
        viewBox="0 0 24 24"
      >
        <path 
          stroke-linecap="round" 
          stroke-linejoin="round" 
          stroke-width="2" 
          d="M19 9l-7 7-7-7"
        />
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'

interface Props {
  modelValue?: string | number
  disabled?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: '',
  disabled: false
})

const emit = defineEmits(['update:modelValue'])

const localValue = ref(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
})

const handleChange = (event: Event) => {
  const target = event.target as HTMLSelectElement
  emit('update:modelValue', target.value)
}
</script>