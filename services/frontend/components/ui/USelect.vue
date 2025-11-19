<script setup lang="ts">
import { ref, watch } from 'vue'

interface Props {
  modelValue?: string | number
  disabled?: boolean
  className?: string
}

interface Emits {
  (e: 'update:modelValue', value: string | number): void
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: '',
  disabled: false
})

const emit = defineEmits<Emits>()

const localValue = ref(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
})

const handleChange = (event: Event) => {
  const target = event.target as HTMLSelectElement
  emit('update:modelValue', target.value)
}
</script>

<template>
  <div class="relative w-full">
    <select v-model="localValue" @change="handleChange" :disabled="props.disabled" :class="[
      'input-metal appearance-none w-full pr-10',
      'cursor-pointer',
      'transition-all duration-200',
      !props.disabled && 'hover:border-primary-400',
      'focus:border-primary-400 focus:ring-2 focus:ring-primary-500/20',
      props.disabled && 'cursor-not-allowed opacity-50 bg-steel-darker',
      props.className
    ]" v-bind="$attrs">
      <slot />
    </select>

    <div :class="[
      'absolute right-3 top-1/2 -translate-y-1/2',
      'pointer-events-none transition-colors duration-200',
      props.disabled ? 'text-steel-dark' : 'text-steel-light',
    ]">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
      </svg>
    </div>
  </div>
</template>
