<template>
  <label 
    :class="[
      'relative flex items-center gap-2 cursor-pointer select-none text-text-primary',
      'hover:text-primary-300 transition-colors',
      disabled && 'cursor-not-allowed opacity-50'
    ]"
  >
    <input
      type="radio"
      :value="value"
      :checked="modelValue === value"
      @input="$emit('update:modelValue', value)"
      class="peer sr-only"
      :disabled="disabled"
    />
    <span
      :class="[
        'relative block w-5 h-5 rounded-full border-2 transition-all duration-200',
        'bg-background-secondary',
        modelValue === value 
          ? 'border-primary-500 shadow-[0_0_0_2px_rgba(79,70,229,0.2)]' 
          : 'border-steel-medium hover:border-steel-light',
        'peer-focus-visible:ring-2 peer-focus-visible:ring-primary-500/30',
        disabled && 'border-steel-dark bg-steel-dark'
      ]"
    >
      <!-- Inner dot -->
      <span
        v-if="modelValue === value"
        class="absolute left-1/2 top-1/2 w-2.5 h-2.5 bg-white rounded-full -translate-x-1/2 -translate-y-1/2 shadow-sm"
      />
    </span>
    <span class="text-sm font-medium">
      <slot />
    </span>
  </label>
</template>

<script setup lang="ts">
interface Props {
  modelValue?: string | number
  value: string | number
  disabled?: boolean
}
const props = defineProps<Props>()
defineEmits(['update:modelValue'])
</script>