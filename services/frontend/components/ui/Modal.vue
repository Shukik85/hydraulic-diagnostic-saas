<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue';

interface ModalProps {
  modelValue: boolean;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closeable?: boolean;
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
}

const props = withDefaults(defineProps<ModalProps>(), {
  size: 'md',
  closeable: true,
  closeOnBackdrop: true,
  closeOnEscape: true,
});

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
  close: [];
}>;

const modalRef = ref<HTMLElement | null>(null);
const previouslyFocusedElement = ref<HTMLElement | null>(null);

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full mx-4',
};

const close = () => {
  if (props.closeable) {
    emit('update:modelValue', false);
    emit('close');
  }
};

const handleBackdropClick = (event: MouseEvent) => {
  if (props.closeOnBackdrop && event.target === event.currentTarget) {
    close();
  }
};

const handleEscapeKey = (event: KeyboardEvent) => {
  if (props.closeOnEscape && event.key === 'Escape') {
    close();
  }
};

const trapFocus = (event: KeyboardEvent) => {
  if (!modalRef.value || event.key !== 'Tab') return;

  const focusableElements = modalRef.value.querySelectorAll<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  if (event.shiftKey) {
    // Shift + Tab
    if (document.activeElement === firstElement) {
      event.preventDefault();
      lastElement?.focus();
    }
  } else {
    // Tab
    if (document.activeElement === lastElement) {
      event.preventDefault();
      firstElement?.focus();
    }
  }
};

watch(() => props.modelValue, async (isOpen) => {
  if (isOpen) {
    // Store currently focused element
    previouslyFocusedElement.value = document.activeElement as HTMLElement;
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
    
    // Focus first focusable element in modal
    await nextTick();
    const firstFocusable = modalRef.value?.querySelector<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    firstFocusable?.focus();
  } else {
    // Restore body scroll
    document.body.style.overflow = '';
    
    // Restore focus to previously focused element
    previouslyFocusedElement.value?.focus();
  }
});

onMounted(() => {
  if (props.closeOnEscape) {
    document.addEventListener('keydown', handleEscapeKey);
  }
  document.addEventListener('keydown', trapFocus);
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleEscapeKey);
  document.removeEventListener('keydown', trapFocus);
  document.body.style.overflow = '';
});
</script>

<template>
  <Teleport to="body">
    <Transition
      enter-active-class="transition ease-out duration-300"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition ease-in duration-200"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="modelValue"
        class="fixed inset-0 z-50 overflow-y-auto"
        aria-labelledby="modal-title"
        aria-modal="true"
        role="dialog"
      >
        <!-- Backdrop -->
        <div
          class="fixed inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
          aria-hidden="true"
          @click="handleBackdropClick"
        />

        <!-- Modal container -->
        <div class="flex min-h-full items-center justify-center p-4">
          <Transition
            enter-active-class="transition ease-out duration-300"
            enter-from-class="opacity-0 scale-95 translate-y-4"
            enter-to-class="opacity-100 scale-100 translate-y-0"
            leave-active-class="transition ease-in duration-200"
            leave-from-class="opacity-100 scale-100 translate-y-0"
            leave-to-class="opacity-0 scale-95 translate-y-4"
          >
            <div
              v-if="modelValue"
              ref="modalRef"
              class="relative w-full bg-white dark:bg-gray-900 rounded-lg shadow-xl transform transition-all"
              :class="sizeClasses[size]"
            >
              <!-- Header -->
              <div
                v-if="title || closeable || $slots.header"
                class="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700"
              >
                <slot name="header">
                  <h3
                    id="modal-title"
                    class="text-lg font-semibold text-gray-900 dark:text-gray-100"
                  >
                    {{ title }}
                  </h3>
                </slot>

                <button
                  v-if="closeable"
                  type="button"
                  class="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 transition-colors"
                  aria-label="Close modal"
                  @click="close"
                >
                  <Icon name="heroicons:x-mark" class="h-6 w-6" aria-hidden="true" />
                </button>
              </div>

              <!-- Body -->
              <div class="px-6 py-4">
                <slot />
              </div>

              <!-- Footer -->
              <div
                v-if="$slots.footer"
                class="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700"
              >
                <slot name="footer" :close="close" />
              </div>
            </div>
          </Transition>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>
