<template>
  <RadioGroupItem
    :value="value"
    :disabled="disabled"
    :class="[
      'group relative flex items-center gap-3 cursor-pointer select-none',
      'text-text-primary font-medium text-sm',
      'transition-all duration-200',
      disabled && 'cursor-not-allowed opacity-50',
      className
    ]"
  >
    <RadioGroupIndicator
      :class="[
        'relative flex items-center justify-center',
        'w-5 h-5 rounded-full',
        'border-2 transition-all duration-200',
        'bg-background-secondary',
        // Unchecked state
        'border-steel-medium group-hover:border-primary-400',
        // Checked state
        'group-data-[state=checked]:border-primary-500',
        'group-data-[state=checked]:bg-primary-500/10',
        'group-data-[state=checked]:shadow-[0_0_8px_rgba(79,70,229,0.3)]',
        // Focus state
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500/30',
        'focus-visible:ring-offset-2 focus-visible:ring-offset-background-primary',
        // Disabled state
        disabled && 'border-steel-dark bg-steel-dark group-hover:border-steel-dark',
      ]"
    >
      <!-- Inner dot (only shows when checked) -->
      <span
        :class="[
          'w-2.5 h-2.5 rounded-full',
          'bg-primary-500 shadow-sm',
          'transition-all duration-200',
          'scale-0 group-data-[state=checked]:scale-100',
        ]"
      />
    </RadioGroupIndicator>
    <label
      :for="value?.toString()"
      :class="[
        'cursor-pointer group-hover:text-primary-300',
        'transition-colors duration-200',
        disabled && 'cursor-not-allowed',
      ]"
    >
      <slot />
    </label>
  </RadioGroupItem>
</template>

<script setup lang="ts">
import { RadioGroupIndicator, RadioGroupItem } from 'radix-vue'

interface Props {
  value: string | number
  disabled?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
})
</script>