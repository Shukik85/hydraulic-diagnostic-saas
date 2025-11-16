<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div 
        v-if="modelValue" 
        class="fixed inset-0 z-50 overflow-y-auto focus:outline-none" 
        aria-modal="true" 
        role="dialog" 
        :aria-labelledby="titleId" 
        @keydown.esc="handleClose" 
        tabindex="0" 
        ref="modalRef"
      >
        <!-- Backdrop (metallic) - Always closes on click -->
        <div 
          class="fixed inset-0 bg-gradient-to-br from-steel-dark/90 to-steel-darker/70 backdrop-blur-md transition-opacity" 
          @click="handleClose" 
        />
        
        <!-- Modal Container (metallic border) -->
        <div class="relative flex min-h-screen items-center justify-center p-4">
          <div 
            class="relative w-full max-w-full transform rounded-2xl bg-gradient-to-br from-steel-dark via-steel bg-opacity-90 shadow-2xl border-[2.5px] border-steel-medium/80 backdrop-blur-xl transition-all duration-200"
            :class="sizeClasses"
            @click.stop
            role="dialog"
            :aria-labelledby="titleId"
            :aria-describedby="description ? descriptionId : undefined"
          >
            <!-- Close Button -->
            <button
              v-if="showCloseButton"
              type="button"
              class="absolute top-4 right-4 text-text-secondary hover:text-white transition-colors rounded-md p-1 hover:bg-steel-dark/50"
              @click="handleClose"
              aria-label="Close modal"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <!-- Header -->
            <div v-if="$slots.header || title" class="border-b border-steel-medium px-6 py-4 bg-steel-darker/70 rounded-t-2xl">
              <slot name="header">
                <div>
                  <h3 :id="titleId" class="text-lg font-bold text-primary-400 tracking-wide uppercase drop-shadow">
                    {{ title }}
                  </h3>
                  <p v-if="description" :id="descriptionId" class="text-sm text-text-secondary mt-1">
                    {{ description }}
                  </p>
                </div>
              </slot>
            </div>

            <!-- Body -->
            <div v-if="$slots.default" class="px-6 py-6 text-text-primary bg-steel/40">
              <slot />
            </div>

            <!-- Footer -->
            <div v-if="$slots.footer" class="flex items-center justify-end gap-3 border-t border-steel-medium bg-steel-darker/70 px-6 py-4 rounded-b-2xl">
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
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  showCloseButton: true
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  close: []
}>()

const modalRef = ref<HTMLElement>()
const titleId = `modal-title-${Math.random().toString(36).substr(2, 9)}`
const descriptionId = `modal-description-${Math.random().toString(36).substr(2, 9)}`

const sizeClasses = computed(() => ({
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full mx-4'
}[props.size]))

const handleClose = () => {
  emit('update:modelValue', false)
  emit('close')
}

const handleEscape = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    handleClose()
  }
}

// Auto-focus modal when opened for accessibility
watch(() => props.modelValue, (isOpen) => {
  if (isOpen) {
    nextTick(() => {
      modalRef.value?.focus()
    })
  }
})
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
</style>
