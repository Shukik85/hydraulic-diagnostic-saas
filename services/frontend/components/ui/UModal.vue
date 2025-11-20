<script setup lang="ts">
import { ref, watch, onUnmounted } from '#imports'
import { useFocusTrap } from '~/composables/useFocusTrap'

interface Props {
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  closeOnBackdrop?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  closeOnBackdrop: true,
})

// ✅ ПРАВИЛЬНЮ: Tuple syntax для emits (Vue 3.5+)
const emit = defineEmits<{
  (e: 'close'): void
}>()

const modalRef = ref<HTMLElement | null>(null)
const titleId = `modal-title-${Math.random().toString(36).substr(2, 9)}`
const descriptionId = `modal-desc-${Math.random().toString(36).substr(2, 9)}`

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full',
}[props.size]

const handleClose = () => {
  emit('close')
}

const onBackdropClick = () => {
  if (props.closeOnBackdrop) {
    handleClose()
  }
}

// Focus trap - только 1 аргумент
const { activate, deactivate } = useFocusTrap(modalRef)

watch(modalRef, (newVal) => {
  if (newVal) {
    activate()
  }
}, { immediate: true })

onUnmounted(() => {
  deactivate()
})
</script>

<template>
  <div class="modal-overlay" @click="onBackdropClick">
    <div
      ref="modalRef"
      role="dialog"
      :aria-labelledby="titleId"
      :aria-describedby="descriptionId"
      aria-modal="true"
      class="modal-content"
      :class="sizeClasses"
      @click.stop
    >
      <slot :title-id="titleId" :description-id="descriptionId" :handle-close="handleClose" />
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  @apply fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50;
}

.modal-content {
  @apply bg-white rounded-lg shadow-xl w-full;
}
</style>
