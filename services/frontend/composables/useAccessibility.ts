import { ref, onMounted, onUnmounted, nextTick, type Ref } from '#imports'

/**
 * A11y Composable - утилиты для доступности
 * Основано на WCAG 2.1 Level AA
 */

interface FocusTrapOptions {
  /** Автоматически закрывать по Escape */
  escapeDeactivates?: boolean
  /** Callback при закрытии */
  onDeactivate?: () => void
}

interface UseAccessibilityReturn {
  /** Сохраненный элемент с фокусом */
  previousActiveElement: Ref<HTMLElement | null>
  /** Активировать focus trap */
  activateFocusTrap: (containerRef: Ref<HTMLElement | null>, options?: FocusTrapOptions) => void
  /** Деактивировать focus trap */
  deactivateFocusTrap: () => void
  /** Восстановить фокус */
  restoreFocus: () => void
  /** Переместить фокус на первый focusable элемент */
  focusFirstElement: (containerRef: Ref<HTMLElement | null>) => void
}

/**
 * Focusable элементы selector
 */
const FOCUSABLE_SELECTOR = [
  'button:not([disabled])',
  '[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])'
].join(', ')

/**
 * Получить все focusable элементы в container
 */
function getFocusableElements(container: HTMLElement | null): HTMLElement[] {
  if (!container) return []
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
}

/**
 * A11y Composable
 */
export function useAccessibility(): UseAccessibilityReturn {
  const previousActiveElement = ref<HTMLElement | null>(null)
  let currentOptions: FocusTrapOptions = {}
  let currentContainerRef: Ref<HTMLElement | null> | null = null
  
  /**
   * Focus trap keyboard handler
   */
  const handleKeydown = (event: KeyboardEvent) => {
    if (!currentContainerRef?.value) return

    // Escape key
    if (event.key === 'Escape' && currentOptions.escapeDeactivates) {
      event.preventDefault()
      deactivateFocusTrap()
      currentOptions.onDeactivate?.()
      return
    }

    // Tab key
    if (event.key === 'Tab') {
      const focusableElements = getFocusableElements(currentContainerRef.value)
      
      if (focusableElements.length === 0) return

      const first = focusableElements[0]
      const last = focusableElements[focusableElements.length - 1]

      // Shift + Tab на первом элементе -> перейти на последний
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last?.focus()
      }
      // Tab на последнем элементе -> перейти на первый
      else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first?.focus()
      }
    }
  }

  /**
   * Активировать focus trap
   */
  const activateFocusTrap = (
    containerRef: Ref<HTMLElement | null>,
    options: FocusTrapOptions = {}
  ) => {
    // Сохраняем текущий фокус
    previousActiveElement.value = document.activeElement as HTMLElement
    
    // Сохраняем опции и ref
    currentOptions = {
      escapeDeactivates: true,
      ...options
    }
    currentContainerRef = containerRef

    // Добавляем event listener
    window.addEventListener('keydown', handleKeydown)

    // Перемещаем фокус на первый элемент
    nextTick(() => {
      focusFirstElement(containerRef)
    })
  }

  /**
   * Деактивировать focus trap
   */
  const deactivateFocusTrap = () => {
    window.removeEventListener('keydown', handleKeydown)
    currentContainerRef = null
    currentOptions = {}
  }

  /**
   * Восстановить фокус
   */
  const restoreFocus = () => {
    nextTick(() => {
      previousActiveElement.value?.focus()
    })
  }

  /**
   * Переместить фокус на первый focusable элемент
   */
  const focusFirstElement = (containerRef: Ref<HTMLElement | null>) => {
    if (!containerRef.value) return

    const focusableElements = getFocusableElements(containerRef.value)
    focusableElements[0]?.focus()
  }

  // Cleanup при unmount
  onUnmounted(() => {
    deactivateFocusTrap()
  })

  return {
    previousActiveElement,
    activateFocusTrap,
    deactivateFocusTrap,
    restoreFocus,
    focusFirstElement
  }
}

/**
 * Утилиты для announce сообщений screen readerам
 */
export function useScreenReaderAnnounce() {
  const message = ref('')
  const announceType = ref<'polite' | 'assertive'>('polite')

  /**
   * Объявить сообщение для screen reader
   */
  const announce = (text: string, type: 'polite' | 'assertive' = 'polite') => {
    message.value = text
    announceType.value = type

    // Очистить сообщение через 5 секунд
    setTimeout(() => {
      message.value = ''
    }, 5000)
  }

  return {
    message,
    announceType,
    announce
  }
}

/**
 * Проверка prefers-reduced-motion
 */
export function usePrefersReducedMotion() {
  const prefersReducedMotion = ref(false)

  onMounted(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    prefersReducedMotion.value = mediaQuery.matches

    const handleChange = (e: MediaQueryListEvent) => {
      prefersReducedMotion.value = e.matches
    }

    mediaQuery.addEventListener('change', handleChange)

    onUnmounted(() => {
      mediaQuery.removeEventListener('change', handleChange)
    })
  })

  return {
    prefersReducedMotion
  }
}
