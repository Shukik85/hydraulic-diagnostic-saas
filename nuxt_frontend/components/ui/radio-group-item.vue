<template>
  <label
    :class="cn(
      'relative flex items-center justify-center w-4 h-4 rounded-full border border-input bg-input-background transition-colors focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
      modelValue === value && 'border-primary',
      className
    )"
  >
    <input
      type="radio"
      :value="value"
      :checked="modelValue === value"
      :disabled="disabled"
      @change="$emit('update:modelValue', value)"
      class="sr-only"
    />
    <div
      v-if="modelValue === value"
      class="absolute inset-0 flex items-center justify-center"
    >
      <div class="w-2 h-2 bg-primary rounded-full" />
    </div>
  </label>
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  value: string;
  modelValue?: string;
  disabled?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  disabled: false,
});

defineEmits<{
  'update:modelValue': [value: string];
}>();
</script>
