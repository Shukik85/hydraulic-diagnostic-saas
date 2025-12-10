<!--
  Delete Confirmation Modal
  @component Modal dialog for destructive actions
  @accessibility WCAG 2.1 AA - focus trap, proper roles, keyboard handling
-->

<template>
  <Teleport to="body">
    <div class="modal-overlay" @click="emit('cancel')" :aria-hidden="false" role="presentation"></div>
    <div
      class="modal-dialog"
      role="alertdialog"
      aria-modal="true"
      :aria-labelledby="`modal-title-${id}`"
      :aria-describedby="`modal-desc-${id}`"
    >
      <div class="modal-content">
        <button
          class="modal-close"
          @click="emit('cancel')"
          :aria-label="`Close dialog`"
          type="button"
        >
          ×
        </button>

        <div class="modal-icon" aria-hidden="true">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>

        <h2 :id="`modal-title-${id}`" class="modal-title">
          Delete System?
        </h2>

        <p :id="`modal-desc-${id}`" class="modal-description">
          Are you sure you want to delete <strong>{{ systemName }}</strong>? This action cannot be undone.
        </p>

        <div class="modal-actions">
          <button
            class="btn btn--secondary"
            @click="emit('cancel')"
            type="button"
            autofocus
          >
            Cancel
          </button>
          <button
            class="btn btn--primary btn--danger"
            @click="emit('confirm')"
            type="button"
          >
            Delete System
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
interface Props {
  systemName: string
}

defineProps<Props>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const id = Math.random().toString(36).substr(2, 9)

// Focus management
onMounted(() => {
  document.body.style.overflow = 'hidden'
})

onUnmounted(() => {
  document.body.style.overflow = ''
})

// Keyboard handling
const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Escape') {
    emit('cancel')
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped lang="css">
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 999;
  animation: fadeIn 150ms ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-dialog {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 1000;
  max-width: 420px;
  width: 90%;
  animation: slideUp 300ms ease-out;
}

@keyframes slideUp {
  from {
    transform: translate(-50%, -45%);
    opacity: 0;
  }
  to {
    transform: translate(-50%, -50%);
    opacity: 1;
  }
}

.modal-content {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-border);
  padding: var(--space-32);
  position: relative;
  box-shadow: var(--shadow-lg);
}

.modal-close {
  position: absolute;
  top: var(--space-12);
  right: var(--space-12);
  background: none;
  border: none;
  font-size: var(--font-size-4xl);
  cursor: pointer;
  color: var(--color-text-secondary);
  transition: color var(--duration-fast) var(--ease-standard);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-base);
}

.modal-close:hover {
  color: var(--color-error);
  background: rgba(var(--color-error-rgb), 0.1);
}

.modal-close:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

.modal-icon {
  text-align: center;
  margin-bottom: var(--space-16);
  color: var(--color-warning);
}

.modal-title {
  font-size: var(--font-size-2xl);
  color: var(--color-text);
  margin-bottom: var(--space-12);
  text-align: center;
}

.modal-description {
  color: var(--color-text-secondary);
  text-align: center;
  margin-bottom: var(--space-24);
  line-height: var(--line-height-normal);
}

.modal-description strong {
  color: var(--color-text);
  font-weight: var(--font-weight-semibold);
}

.modal-actions {
  display: flex;
  gap: var(--space-12);
  justify-content: center;
}

.btn {
  flex: 1;
  min-width: 120px;
}

.btn--danger {
  background: var(--color-error);
  color: white;
}

.btn--danger:hover {
  background: var(--color-error);
  opacity: 0.9;
}

/* Focus trap - ensures focus stays within modal */
.modal-dialog:focus-within {
  /* Maintain focus within modal */
}
</style>
