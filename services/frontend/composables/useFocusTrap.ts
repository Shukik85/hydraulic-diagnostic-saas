/**
 * useFocusTrap.ts - A11y Focus Trap для modal окон
 * 
 * Использование:
 * ```typescript
 * const modalRef = ref<HTMLElement | null>(null)
 * const { activate, deactivate } = useFocusTrap(modalRef)
 * 
 * // При открытии modal
 * activate()
 * 
 * // При закрытии
 * deactivate()
 * ```
 */
import { ref, watch, onBeforeUnmount, type Ref } from 'vue'

export function useFocusTrap(containerRef: Ref<HTMLElement | null>) {
  const isActive = ref(false)
  let cleanup: (() => void) | null = null

  const focusableSelector = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',')

  function activate() {
    if (!containerRef.value) {
      console.warn('useFocusTrap: containerRef is null')
      return
    }

    isActive.value = true
    
    const focusableElements = containerRef.value.querySelectorAll<HTMLElement>(focusableSelector)
    if (focusableElements.length === 0) {
      console.warn('useFocusTrap: no focusable elements found')
      return
    }
    
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]
    
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== 'Tab') return
      
      if (event.shiftKey) {
        // Shift+Tab: переход назад
        if (document.activeElement === firstElement) {
          event.preventDefault()
          lastElement?.focus()
        }
      } else {
        // Tab: переход вперёд
        if (document.activeElement === lastElement) {
          event.preventDefault()
          firstElement?.focus()
        }
      }
    }
    
    containerRef.value.addEventListener('keydown', handleKeyDown)
    
    // Сохраняем cleanup функцию
    cleanup = () => {
      containerRef.value?.removeEventListener('keydown', handleKeyDown)
    }
    
    // Фокус на первом элементе
    firstElement?.focus()
  }

  function deactivate() {
    isActive.value = false
    cleanup?.()
    cleanup = null
  }

  // Auto-activate при появлении containerRef
  watch(() => containerRef.value, (newVal) => {
    if (newVal && isActive.value) {
      activate()
    }
  })

  // Cleanup при unmount
  onBeforeUnmount(() => {
    deactivate()
  })

  return {
    isActive,
    activate,
    deactivate
  }
}
