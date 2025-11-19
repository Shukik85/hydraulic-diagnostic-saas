<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        ref="modalRef"
        class="fixed inset-0 z-50 overflow-y-auto"
        aria-modal="true" 
        role="dialog" 
        :aria-labelledby="titleId" 
        :aria-describedby="description ? descriptionId : undefined"
      >
        <!-- Backdrop (metallic) -->
        <div 
          class="fixed inset-0 bg-gradient-to-br from-steel-dark/90 to-steel-darker/70 backdrop-blur-md transition-opacity" 
          @click="onBackdropClick"
          aria-hidden="true"
        />
        
        <!-- Modal Container (metallic border) -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full max-w-full transform rounded-2xl bg-gradient-to-br from-steel-dark via-steel bg-opacity-90 shadow-2xl border-[2.5px] border-steel-medium/80 backdrop-blur-xl transition-all duration-200"
            :class="sizeClasses"
            @click.stop
          >
            <!-- Close button for keyboard users -->
            <button
              v-if="closeOnBackdrop"
              type="button"
              @click="handleClose"
              class="absolute top-4 right-4 p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-steel-darker/50 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-steel-dark"
              :aria-label="$t('ui.closeModal', 'Закрыть модальное окно')"
            >
              <Icon name="heroicons:x-mark" class="w-5 h-5" aria-hidden="true" />
            </button>

            <!-- Header -->
            <div 
              v-if="$slots.header || title" 
              class="border-b border-steel-medium px-6 py-4 bg-steel-darker/70 rounded-t-2xl"
            >
              <slot name="header">
                <div>
                  <h2 
                    :id="titleId" 
                    class="text-lg font-bold text-primary-400 tracking-wide uppercase drop-shadow"
                  >
                    {{ title }}
                  </h2>
                  <p 
                    v-if="description" 
                    :id="descriptionId" 
                    class="text-sm text-text-secondary mt-1"
                  >
                    {{ description }}
                  </p>
                </div>
              </slot>
            </div>
            
            <!-- Body -->
            <div 
              v-if="$slots.default" 
              class="px-6 py-6 text-text-primary bg-steel/40"
            >
              <slot />
            </div>
            
            <!-- Footer -->
            <div 
              v-if="$slots.footer" 
              class="flex items-center justify-end gap-3 border-t border-steel-medium bg-steel-darker/70 px-6 py-4 rounded-b-2xl"
            >
              <slot name="footer" />
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, ref, watch, nextTick } from '#imports'
import type { Ref } from 'vue'

interface Props {
  modelValue: boolean
  title?: string
  description?: string
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  closeOnBackdrop: true,
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'close': []
}>()

const modalRef: Ref<HTMLElement | null> = ref(null)

// Generate unique IDs for accessibility
const { generateId, announceMessage } = useA11y()
const titleId = generateId('modal-title')
const descriptionId = generateId('modal-description')

// Setup focus trap
const { activate, deactivate } = useFocusTrap(modalRef, {
  escapeDeactivates: true,
  onEscape: handleClose,
})

// Size classes
const sizeClasses = computed(() => ({
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full mx-4',
}[props.size]))

function handleClose() {
  emit('update:modelValue', false)
  emit('close')
}

function onBackdropClick() {
  if (props.closeOnBackdrop) {
    handleClose()
  }
}

// Watch modal state and manage focus
watch(
  () => props.modelValue,
  async (isOpen) => {
    if (isOpen) {
      // Wait for modal to be rendered
      await nextTick()
      
      // Activate focus trap
      activate()
      
      // Announce modal opening to screen readers
      if (props.title) {
        announceMessage(
          `${props.title}. ${props.description || ''}`,
          'polite'
        )
      }
      
      // Prevent body scroll
      document.body.style.overflow = 'hidden'
    } else {
      // Deactivate focus trap
      deactivate()
      
      // Restore body scroll
      document.body.style.overflow = ''
    }
  }
)
</script>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: all 0.25s ease-out;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
  transform: scale(0.95);
}

/* Reduce motion for users who prefer it */
@media (prefers-reduced-motion: reduce) {
  .modal-enter-active,
  .modal-leave-active {
    transition: none;
  }
}
</style>
