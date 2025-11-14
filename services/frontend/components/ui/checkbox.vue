<template>
  <label
    :class="
      cn(
        'relative flex items-center justify-center w-4 h-4 rounded border transition-colors cursor-pointer',
        'border-steel-medium bg-background-secondary',
        'focus-within:ring-2 focus-within:ring-primary-500/30',
        'hover:border-primary-400',
        'disabled:cursor-not-allowed disabled:opacity-50',
        modelValue && 'bg-primary-500 border-primary-500',
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
    <Icon v-if="modelValue" name="lucide:check" class="h-3.5 w-3.5 text-white" />
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