<script setup lang="ts" generic="T extends string | number">
import { computed } from 'vue';

interface RadioOption<T> {
  label: string;
  value: T;
  description?: string;
  disabled?: boolean;
}

interface RadioProps<T> {
  modelValue?: T;
  options: RadioOption<T>[];
  name: string;
  label?: string;
  error?: string;
  disabled?: boolean;
  layout?: 'vertical' | 'horizontal';
}

const props = withDefaults(defineProps<RadioProps<T>>(), {
  disabled: false,
  layout: 'vertical',
});

const emit = defineEmits<{
  'update:modelValue': [value: T];
  change: [value: T];
}>();

const radioGroupId = computed(() => useId());

const handleChange = (value: T) => {
  emit('update:modelValue', value);
  emit('change', value);
};

const isChecked = (value: T) => {
  return props.modelValue === value;
};
</script>

<template>
  <div class="space-y-2">
    <label
      v-if="label"
      :id="radioGroupId"
      class="block text-sm font-medium text-gray-900 dark:text-gray-100"
    >
      {{ label }}
    </label>

    <div
      role="radiogroup"
      :aria-labelledby="label ? radioGroupId : undefined"
      :aria-invalid="!!error"
      :class="[
        layout === 'horizontal' ? 'flex flex-wrap gap-4' : 'space-y-3',
      ]"
    >
      <div
        v-for="option in options"
        :key="String(option.value)"
        class="flex items-start"
      >
        <div class="flex items-center h-5">
          <input
            :id="`${name}-${String(option.value)}`"
            type="radio"
            :name="name"
            :value="option.value"
            :checked="isChecked(option.value)"
            :disabled="disabled || option.disabled"
            :aria-describedby="option.description ? `${name}-${String(option.value)}-description` : undefined"
            class="h-4 w-4 border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
            :class="[
              error ? 'border-red-500' : '',
              disabled || option.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
            ]"
            @change="handleChange(option.value)"
          />
        </div>

        <div class="ml-3">
          <label
            :for="`${name}-${String(option.value)}`"
            class="text-sm font-medium text-gray-900 dark:text-gray-100"
            :class="disabled || option.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'"
          >
            {{ option.label }}
          </label>

          <p
            v-if="option.description"
            :id="`${name}-${String(option.value)}-description`"
            class="text-sm text-gray-500 dark:text-gray-400"
          >
            {{ option.description }}
          </p>
        </div>
      </div>
    </div>

    <!-- Error message -->
    <p
      v-if="error"
      class="text-sm text-red-600 dark:text-red-400"
      role="alert"
    >
      {{ error }}
    </p>
  </div>
</template>
