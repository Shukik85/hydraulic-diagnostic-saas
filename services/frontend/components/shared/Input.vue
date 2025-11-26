<script setup lang="ts">
import { computed, ref, watch } from 'vue';

interface Props {
  modelValue: string | number;
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url';
  placeholder?: string;
  label?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  icon?: string;
  autocomplete?: string;
}

const props = withDefaults(defineProps<Props>(), {
  type: 'text',
  disabled: false,
  required: false,
});

interface Emits {
  (e: 'update:modelValue', value: string | number): void;
  (e: 'blur', event: FocusEvent): void;
  (e: 'focus', event: FocusEvent): void;
}

const emit = defineEmits<Emits>();

const inputId = ref(`input-${Math.random().toString(36).substr(2, 9)}`);
const isFocused = ref(false);

const inputClasses = computed(() => {
  const classes = [
    'w-full',
    'px-4',
    'py-2',
    'border',
    'rounded-lg',
    'transition-colors',
    'focus:outline-none',
    'focus:ring-2',
    'disabled:opacity-50',
    'disabled:cursor-not-allowed',
  ];

  if (props.error) {
    classes.push(
      'border-red-500',
      'focus:border-red-500',
      'focus:ring-red-500',
      'dark:border-red-400'
    );
  } else {
    classes.push(
      'border-gray-300',
      'focus:border-primary-500',
      'focus:ring-primary-500',
      'dark:border-gray-600',
      'dark:bg-gray-800',
      'dark:text-gray-100'
    );
  }

  if (props.icon) {
    classes.push('pl-10');
  }

  return classes.join(' ');
});

const handleInput = (event: Event): void => {
  const target = event.target as HTMLInputElement;
  const value = props.type === 'number' ? Number(target.value) : target.value;
  emit('update:modelValue', value);
};

const handleBlur = (event: FocusEvent): void => {
  isFocused.value = false;
  emit('blur', event);
};

const handleFocus = (event: FocusEvent): void => {
  isFocused.value = true;
  emit('focus', event);
};
</script>

<template>
  <div class="w-full">
    <label v-if="label" :for="inputId" class="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
      {{ label }}
      <span v-if="required" class="text-red-500" aria-label="required">*</span>
    </label>

    <div class="relative">
      <Icon v-if="icon" :name="icon" class="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400" aria-hidden="true" />

      <input
        :id="inputId"
        :type="type"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :required="required"
        :autocomplete="autocomplete"
        :class="inputClasses"
        :aria-invalid="!!error"
        :aria-describedby="error ? `${inputId}-error` : undefined"
        @input="handleInput"
        @blur="handleBlur"
        @focus="handleFocus"
      />
    </div>

    <p v-if="error" :id="`${inputId}-error`" class="mt-1 text-sm text-red-600 dark:text-red-400" role="alert">
      {{ error }}
    </p>
  </div>
</template>
