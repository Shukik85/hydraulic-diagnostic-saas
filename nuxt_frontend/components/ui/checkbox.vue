<template>
  <label
    :class="
      cn(
        'relative flex items-center justify-center w-4 h-4 rounded border border-input bg-input-background transition-colors focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
        modelValue && 'bg-primary border-primary',
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
    <Icon v-if="modelValue" name="lucide:check" class="h-3.5 w-3.5 text-primary-foreground" />
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
