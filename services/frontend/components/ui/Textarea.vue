<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';

interface TextareaProps {
  modelValue?: string;
  label?: string;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  readonly?: boolean;
  rows?: number;
  maxlength?: number;
  autoResize?: boolean;
  showCounter?: boolean;
  id?: string;
}

const props = withDefaults(defineProps<TextareaProps>(), {
  disabled: false,
  required: false,
  readonly: false,
  rows: 3,
  autoResize: false,
  showCounter: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: string];
  blur: [event: FocusEvent];
  focus: [event: FocusEvent];
}>;

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const textareaId = computed(() => props.id || useId());

const characterCount = computed(() => {
  return props.modelValue?.length || 0;
});

const isOverLimit = computed(() => {
  if (!props.maxlength) return false;
  return characterCount.value > props.maxlength;
});

const handleInput = (event: Event) => {
  const target = event.target as HTMLTextAreaElement;
  emit('update:modelValue', target.value);

  if (props.autoResize) {
    resize();
  }
};

const handleBlur = (event: FocusEvent) => {
  emit('blur', event);
};

const handleFocus = (event: FocusEvent) => {
  emit('focus', event);
};

const resize = () => {
  if (!textareaRef.value || !props.autoResize) return;

  textareaRef.value.style.height = 'auto';
  textareaRef.value.style.height = `${textareaRef.value.scrollHeight}px`;
};

watch(() => props.modelValue, async () => {
  if (props.autoResize) {
    await nextTick();
    resize();
  }
});
</script>

<template>
  <div class="space-y-2">
    <label
      v-if="label"
      :for="textareaId"
      class="block text-sm font-medium text-gray-900 dark:text-gray-100"
    >
      {{ label }}
      <span v-if="required" class="text-red-500 ml-1" aria-label="required">*</span>
    </label>

    <div class="relative">
      <textarea
        :id="textareaId"
        ref="textareaRef"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        :required="required"
        :rows="rows"
        :maxlength="maxlength"
        :aria-invalid="!!error"
        :aria-describedby="error ? `${textareaId}-error` : undefined"
        class="flex w-full rounded-md border bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 ring-offset-white transition-colors resize-y"
        :class="[
          error
            ? 'border-red-500 focus-visible:ring-red-500'
            : 'border-gray-300 dark:border-gray-600 focus-visible:ring-primary-500',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'read-only:cursor-default read-only:bg-gray-50 dark:read-only:bg-gray-900',
          autoResize ? 'resize-none overflow-hidden' : '',
        ]"
        @input="handleInput"
        @blur="handleBlur"
        @focus="handleFocus"
      />
    </div>

    <!-- Counter & Error row -->
    <div class="flex items-center justify-between">
      <p
        v-if="error"
        :id="`${textareaId}-error`"
        class="text-sm text-red-600 dark:text-red-400"
        role="alert"
      >
        {{ error }}
      </p>
      <div v-else />

      <p
        v-if="showCounter && maxlength"
        class="text-sm"
        :class="isOverLimit ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'"
      >
        {{ characterCount }} / {{ maxlength }}
      </p>
    </div>
  </div>
</template>
