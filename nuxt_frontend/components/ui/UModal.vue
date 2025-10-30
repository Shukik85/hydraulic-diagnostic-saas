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
        tabindex="0"
        ref="modalRef"
      >
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/50 transition-opacity" />
        
        <!-- Modal Container -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full transform rounded-xl bg-white shadow-2xl transition-all"
            :class="sizeClasses"
            @click.stop
            role="dialog"
            :aria-labelledby="titleId"
            :aria-describedby="description ? descriptionId : undefined"
          >
            <!-- Header -->
            <div v-if="$slots.header || title" class="border-b border-gray-200 px-6 py-4">
              <slot name="header">
                <div>
                  <h3 :id="titleId" class="text-lg font-semibold text-gray-900">
                    {{ title }}
                  </h3>
                  <p v-if="description" :id="descriptionId" class="text-sm text-gray-500 mt-1">
                    {{ description }}
                  </p>
                </div>
              </slot>
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
  closeOnBackdrop?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  closeOnBackdrop: true
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'close': []
}>()

const modalRef = ref<HTMLElement>()

// Generate unique IDs for accessibility
const titleId = `modal-title-${Math.random().toString(36).substr(2, 9)}`
const descriptionId = `modal-description-${Math.random().toString(36).substr(2, 9)}`

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