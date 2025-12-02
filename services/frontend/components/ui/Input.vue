<script setup lang="ts">
import { computed } from 'vue';

interface InputProps {
  modelValue?: string | number;
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search';
  label?: string;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  readonly?: boolean;
  autocomplete?: string;
  icon?: string;
  id?: string;
}

const props = withDefaults(defineProps<InputProps>(), {
  type: 'text',
  disabled: false,
  required: false,
  readonly: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: string | number];
  blur: [event: FocusEvent];
  focus: [event: FocusEvent];
}>();

const inputId = computed(() => props.id || `input-${Math.random().toString(36).substr(2, 9)}`);

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement;
  emit('update:modelValue', target.value);
};

const handleBlur = (event: FocusEvent) => {
  emit('blur', event);
};

const handleFocus = (event: FocusEvent) => {
  emit('focus', event);
};
</script>

<template>
  <div class="space-y-2">
    <label
      v-if="label"
      :for="inputId"
      class="block text-sm font-medium text-foreground"
    >
      {{ label }}
      <span v-if="required" class="text-destructive ml-1" aria-label="required">*</span>
    </label>

    <div class="relative">
      <div v-if="icon" class="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
        <Icon :name="icon" class="h-5 w-5" />
      </div>

      <input
        :id="inputId"
        :type="type"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        :required="required"
        :autocomplete="autocomplete"
        :aria-invalid="!!error"
        :aria-describedby="error ? `${inputId}-error` : undefined"
        class="flex h-10 w-full rounded-md border bg-background px-3 py-2 text-sm ring-offset-background transition-colors"
        :class="[
          icon ? 'pl-10' : '',
          error
            ? 'border-destructive focus-visible:ring-destructive'
            : 'border-input focus-visible:ring-ring',
          'file:border-0 file:bg-transparent file:text-sm file:font-medium',
          'placeholder:text-muted-foreground',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'read-only:cursor-default read-only:bg-muted',
        ]"
        @input="handleInput"
        @blur="handleBlur"
        @focus="handleFocus"
      />
    </div>

    <p
      v-if="error"
      :id="`${inputId}-error`"
      class="text-sm text-destructive"
      role="alert"
    >
      {{ error }}
    </p>
  </div>
</template>
