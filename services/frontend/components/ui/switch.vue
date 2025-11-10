<template>
  <label
    :class="
      cn(
        'relative inline-flex h-6 w-11 cursor-pointer items-center rounded-full border border-transparent bg-gray-200 transition-colors focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2',
        modelValue && 'bg-primary',
        disabled && 'cursor-not-allowed opacity-50',
        className
      )
    "
  >
    <input
      type="checkbox"
      :checked="modelValue"
      :disabled="disabled"
      @change="$emit('update:modelValue', ($event.target as HTMLInputElement).checked)"
      class="sr-only"
    />
    <span
      :class="
        cn(
          'inline-block h-5 w-5 transform rounded-full bg-white transition-transform',
          modelValue ? 'translate-x-6' : 'translate-x-1'
        )
      "
    />
  </label>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  modelValue?: boolean;
  disabled?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  modelValue: false,
  disabled: false,
});

defineEmits<{
  'update:modelValue': [value: boolean];
}>();
</script>
