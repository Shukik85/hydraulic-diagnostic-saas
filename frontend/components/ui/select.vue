<template>
  <div class="relative">
    <button
      ref="triggerRef"
      :class="
        cn(
          'flex w-full items-center justify-between gap-2 rounded-md border bg-input-background px-3 py-2 text-sm transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
          size === 'sm' ? 'h-8' : 'h-9',
          className
        )
      "
      @click="toggle"
      v-bind="$attrs"
    >
      <span v-if="modelValue" class="flex items-center gap-2">
        <slot name="selected" :value="modelValue">{{ modelValue }}</slot>
      </span>
      <span v-else class="text-muted-foreground">
        <slot name="placeholder">{{ placeholder }}</slot>
      </span>
      <Icon name="lucide:chevron-down" class="h-4 w-4 opacity-50" />
    </button>
    <Transition
      enter-active-class="transition-all duration-200"
      enter-from-class="opacity-0 scale-95"
      enter-to-class="opacity-100 scale-100"
      leave-active-class="transition-all duration-200"
      leave-from-class="opacity-100 scale-100"
      leave-to-class="opacity-0 scale-95"
    >
      <div
        v-if="isOpen"
        ref="contentRef"
        :class="
          cn(
            'absolute z-50 bg-popover text-popover-foreground rounded-md border shadow-md p-1 min-w-32 max-h-60 overflow-y-auto',
            contentClass
          )
        "
        :style="{ top: position.top + 'px', left: position.left + 'px' }"
      >
        <slot />
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { cn } from './utils';

interface Props {
  modelValue?: string;
  placeholder?: string;
  size?: 'sm' | 'default';
  className?: string;
  contentClass?: string;
}

const props = withDefaults(defineProps<Props>(), {
  size: 'default',
  placeholder: 'Select...',
});

const emit = defineEmits<{
  'update:modelValue': [value: string];
}>();

const isOpen = ref(false);
const triggerRef = ref<HTMLElement>();
const contentRef = ref<HTMLElement>();
const position = ref({ top: 0, left: 0 });

const toggle = () => {
  isOpen.value = !isOpen.value;
  if (isOpen.value) {
    updatePosition();
  }
};

const updatePosition = () => {
  if (triggerRef.value) {
    const rect = triggerRef.value.getBoundingClientRect();
    position.value = {
      top: rect.bottom,
      left: rect.left,
    };
  }
};

const select = (value: string) => {
  emit('update:modelValue', value);
  isOpen.value = false;
};

const close = () => {
  isOpen.value = false;
};

const handleClickOutside = (event: Event) => {
  if (
    triggerRef.value &&
    !triggerRef.value.contains(event.target as Node) &&
    contentRef.value &&
    !contentRef.value.contains(event.target as Node)
  ) {
    close();
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
});

defineExpose({
  select,
});
</script>
