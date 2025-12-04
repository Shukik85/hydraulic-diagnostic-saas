<script setup lang="ts">
import { computed } from 'vue';

interface CheckboxProps {
  modelValue?: boolean;
  label?: string;
  description?: string;
  error?: string;
  disabled?: boolean;
  indeterminate?: boolean;
  id?: string;
}

const props = withDefaults(defineProps<CheckboxProps>(), {
  modelValue: false,
  disabled: false,
  indeterminate: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
  change: [value: boolean];
}>;

const checkboxId = computed(() => props.id || useId());

const handleChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  emit('update:modelValue', target.checked);
  emit('change', target.checked);
};
</script>

<template>
  <div class="flex items-start">
    <div class="flex items-center h-5">
      <input
        :id="checkboxId"
        type="checkbox"
        :checked="modelValue"
        :indeterminate="indeterminate"
        :disabled="disabled"
        :aria-invalid="!!error"
        :aria-describedby="error ? `${checkboxId}-error` : description ? `${checkboxId}-description` : undefined"
        class="h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
        :class="[
          error ? 'border-red-500' : '',
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
        ]"
        @change="handleChange"
      />
    </div>
    
    <div v-if="label || description" class="ml-3">
      <label
        :for="checkboxId"
        class="text-sm font-medium text-gray-900 dark:text-gray-100"
        :class="disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'"
      >
        {{ label }}
      </label>
      
      <p
        v-if="description"
        :id="`${checkboxId}-description`"
        class="text-sm text-gray-500 dark:text-gray-400"
      >
        {{ description }}
      </p>
      
      <p
        v-if="error"
        :id="`${checkboxId}-error`"
        class="mt-1 text-sm text-red-600 dark:text-red-400"
        role="alert"
      >
        {{ error }}
      </p>
    </div>
  </div>
</template>
