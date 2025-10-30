<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        class="fixed inset-0 z-50 overflow-y-auto" 
        aria-modal="true" 
        role="dialog" 
        :aria-labelledby="titleId"
        @click="onBackdropClick"
        @keydown.esc="handleEscape"
      >
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/50 transition-opacity" />
        
        <!-- Modal Container -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full transform rounded-xl bg-white shadow-2xl transition-all"
            :class="sizeClasses"
            @click.stop
            ref="modalRef"
          >
            <!-- Header -->
            <div v-if="$slots.header || title" class="flex items-center justify-between border-b border-gray-200 px-6 py-4">
              <slot name="header">
                <div>
                  <h3 :id="titleId" class="text-lg font-semibold text-gray-900">
                    {{ title }}
                  </h3>
                  <p v-if="description" class="text-sm text-gray-500 mt-1">
                    {{ description }}
                  </p>
                </div>
              </slot>
              
              <button 
                v-if="showCloseButton"
                class="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
                @click="handleClose"
                :disabled="loading"
                :aria-label="$t('ui.close')"
              >
                <Icon name="heroicons:x-mark" class="h-5 w-5" />
              </button>
            </div>

            <!-- Body -->
            <div v-if="$slots.default" class="px-6 py-6">
              <slot />
            </div>

            <!-- Footer -->
            <div v-if="$slots.footer" class="flex items-center justify-end gap-3 border-t border-gray-200 px-6 py-4">
              <slot name="footer" />
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

interface Props {
  modelValue: boolean
  title?: string
  description?: string
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
  closeOnBackdrop?: boolean
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  showCloseButton: true,
  closeOnBackdrop: true,
  loading: false
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'close': []
}>()

const modalRef = ref<HTMLElement>()

// Generate unique ID for accessibility
const titleId = `modal-title-${Math.random().toString(36).substr(2, 9)}`

// Size classes mapping
const sizeClasses = computed(() => {
  const sizeMap = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-full mx-4'
  }
  return sizeMap[props.size]
})

// Event handlers
const handleClose = () => {
  if (props.loading) return
  emit('update:modelValue', false)
  emit('close')
}

const handleEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    handleClose()
  }
}

const onBackdropClick = (event: Event) => {
  if (event.target === event.currentTarget && props.closeOnBackdrop) {
    handleClose()
  }
}

// Focus management
watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    nextTick(() => {
      modalRef.value?.focus()
    })
  }
})
</script>

<style scoped>
/* Modal transitions */
.modal-enter-active,
.modal-leave-active {
  transition: all 0.25s ease-out;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>