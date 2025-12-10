<script setup lang="ts" generic="T extends string | number">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue';

interface SelectOption<T> {
  label: string;
  value: T;
  disabled?: boolean;
}

interface SelectProps<T> {
  modelValue?: T | T[] | null;
  options: SelectOption<T>[];
  label?: string;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  multiple?: boolean;
  searchable?: boolean;
  id?: string;
}

const props = withDefaults(defineProps<SelectProps<T>>(), {
  disabled: false,
  required: false,
  multiple: false,
  searchable: false,
  placeholder: 'Select an option',
});

const emit = defineEmits<{
  'update:modelValue': [value: T | T[] | null];
  change: [value: T | T[] | null];
}>;

const isOpen = ref(false);
const searchQuery = ref('');
const selectRef = ref<HTMLElement | null>(null);
const selectId = computed(() => props.id || useId());
const highlightedIndex = ref(-1);

const selectedOptions = computed(() => {
  if (!props.modelValue) return [];
  
  const values = Array.isArray(props.modelValue) ? props.modelValue : [props.modelValue];
  return props.options.filter(opt => values.includes(opt.value));
});

const displayText = computed(() => {
  if (selectedOptions.value.length === 0) return props.placeholder;
  if (selectedOptions.value.length === 1) return selectedOptions.value[0].label;
  return `${selectedOptions.value.length} selected`;
});

const filteredOptions = computed(() => {
  if (!props.searchable || !searchQuery.value) return props.options;
  
  const query = searchQuery.value.toLowerCase();
  return props.options.filter(opt => 
    opt.label.toLowerCase().includes(query)
  );
});

const toggleDropdown = () => {
  if (props.disabled) return;
  isOpen.value = !isOpen.value;
  if (isOpen.value) {
    highlightedIndex.value = -1;
  }
};

const selectOption = (option: SelectOption<T>) => {
  if (option.disabled) return;

  let newValue: T | T[] | null;

  if (props.multiple) {
    const currentValues = Array.isArray(props.modelValue) ? [...props.modelValue] : [];
    const index = currentValues.findIndex(v => v === option.value);
    
    if (index > -1) {
      currentValues.splice(index, 1);
    } else {
      currentValues.push(option.value);
    }
    
    newValue = currentValues;
  } else {
    newValue = option.value;
    isOpen.value = false;
  }

  emit('update:modelValue', newValue);
  emit('change', newValue);
};

const isSelected = (option: SelectOption<T>) => {
  if (!props.modelValue) return false;
  
  if (Array.isArray(props.modelValue)) {
    return props.modelValue.includes(option.value);
  }
  
  return props.modelValue === option.value;
};

const handleKeydown = (event: KeyboardEvent) => {
  if (!isOpen.value) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleDropdown();
    }
    return;
  }

  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault();
      highlightedIndex.value = Math.min(highlightedIndex.value + 1, filteredOptions.value.length - 1);
      break;
    case 'ArrowUp':
      event.preventDefault();
      highlightedIndex.value = Math.max(highlightedIndex.value - 1, 0);
      break;
    case 'Enter':
      event.preventDefault();
      if (highlightedIndex.value >= 0) {
        selectOption(filteredOptions.value[highlightedIndex.value]);
      }
      break;
    case 'Escape':
      event.preventDefault();
      isOpen.value = false;
      break;
  }
};

const handleClickOutside = (event: MouseEvent) => {
  if (selectRef.value && !selectRef.value.contains(event.target as Node)) {
    isOpen.value = false;
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
});

watch(isOpen, (open) => {
  if (open) {
    searchQuery.value = '';
  }
});
</script>

<template>
  <div ref="selectRef" class="select-container">
    <label
      v-if="label"
      :for="selectId"
      class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2"
    >
      {{ label }}
      <span v-if="required" class="text-red-500 ml-1" aria-label="required">*</span>
    </label>

    <div class="relative">
      <button
        :id="selectId"
        type="button"
        :disabled="disabled"
        :aria-expanded="isOpen"
        :aria-haspopup="true"
        :aria-invalid="!!error"
        :aria-describedby="error ? `${selectId}-error` : undefined"
        class="w-full flex items-center justify-between px-3 py-2 bg-white dark:bg-gray-800 border rounded-md text-left transition-colors"
        :class="[
          error
            ? 'border-red-500 focus:ring-red-500'
            : 'border-gray-300 dark:border-gray-600 focus:ring-primary-500',
          'focus:outline-none focus:ring-2 focus:ring-offset-2',
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
        ]"
        @click="toggleDropdown"
        @keydown="handleKeydown"
      >
        <span
          class="block truncate"
          :class="selectedOptions.length === 0 ? 'text-gray-400' : 'text-gray-900 dark:text-gray-100'"
        >
          {{ displayText }}
        </span>
        <Icon
          name="heroicons:chevron-down"
          class="h-5 w-5 text-gray-400 transition-transform"
          :class="{ 'rotate-180': isOpen }"
          aria-hidden="true"
        />
      </button>

      <!-- Dropdown -->
      <Transition
        enter-active-class="transition ease-out duration-100"
        enter-from-class="opacity-0 scale-95"
        enter-to-class="opacity-100 scale-100"
        leave-active-class="transition ease-in duration-75"
        leave-from-class="opacity-100 scale-100"
        leave-to-class="opacity-0 scale-95"
      >
        <div
          v-if="isOpen"
          class="absolute z-10 mt-1 w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-auto"
          role="listbox"
          :aria-labelledby="selectId"
        >
          <!-- Search input -->
          <div v-if="searchable" class="p-2 border-b border-gray-200 dark:border-gray-700">
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search..."
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              @click.stop
            />
          </div>

          <!-- Options -->
          <div v-if="filteredOptions.length > 0" class="py-1">
            <button
              v-for="(option, index) in filteredOptions"
              :key="String(option.value)"
              type="button"
              :disabled="option.disabled"
              :aria-selected="isSelected(option)"
              :aria-disabled="option.disabled"
              role="option"
              class="w-full flex items-center px-3 py-2 text-left transition-colors"
              :class="[
                highlightedIndex === index ? 'bg-gray-100 dark:bg-gray-700' : '',
                isSelected(option) ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300' : 'text-gray-900 dark:text-gray-100',
                option.disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer',
              ]"
              @click="selectOption(option)"
            >
              <Icon
                v-if="multiple"
                :name="isSelected(option) ? 'heroicons:check-circle' : 'heroicons:circle'"
                class="h-5 w-5 mr-2"
                :class="isSelected(option) ? 'text-primary-600' : 'text-gray-300'"
                aria-hidden="true"
              />
              <span class="block truncate">{{ option.label }}</span>
              <Icon
                v-if="!multiple && isSelected(option)"
                name="heroicons:check"
                class="ml-auto h-5 w-5 text-primary-600"
                aria-hidden="true"
              />
            </button>
          </div>

          <!-- No results -->
          <div v-else class="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">
            No options found
          </div>
        </div>
      </Transition>
    </div>

    <!-- Error message -->
    <p
      v-if="error"
      :id="`${selectId}-error`"
      class="mt-2 text-sm text-red-600 dark:text-red-400"
      role="alert"
    >
      {{ error }}
    </p>
  </div>
</template>

<style scoped>
.select-container {
  @apply w-full;
}
</style>
