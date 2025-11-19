/**
 * Composable для реализации focus trap в модальных окнах
 * Удерживает фокус внутри контейнера и возвращает его при закрытии
 */

import { ref, onMounted, onUnmounted } from '#imports'
import type { Ref } from 'vue'

interface UseFocusTrapOptions {
  /** Начальный элемент для фокуса */
  initialFocus?: string
  /** Включить обработку Escape */
  escapeDeactivates?: boolean
  /** Callback при нажатии Escape */
  onEscape?: () => void
}

export const useFocusTrap = (
  containerRef: Ref<HTMLElement | null>,
  options: UseFocusTrapOptions = {}
) => {
  const { initialFocus, escapeDeactivates = true, onEscape } = options

  const previousActiveElement: Ref<HTMLElement | null> = ref(null)
  const isActive = ref(false)

  /**
   * Получает все фокусируемые элементы внутри контейнера
   */
  const getFocusableElements = (): HTMLElement[] => {
    if (!containerRef.value) return []

    const focusableSelectors = [
      'button:not([disabled])',
      '[href]',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ')

    return Array.from(
      containerRef.value.querySelectorAll<HTMLElement>(focusableSelectors)
    )
  }

  /**
   * Активирует focus trap
   */
  const activate = () => {
    if (!containerRef.value) return

    // Сохраняем текущий фокус
    previousActiveElement.value = document.activeElement as HTMLElement
    isActive.value = true

    // Устанавливаем фокус на начальный элемент
    const focusableElements = getFocusableElements()

    if (initialFocus) {
      const targetElement = containerRef.value.querySelector<HTMLElement>(initialFocus)
      targetElement?.focus()
    } else if (focusableElements.length > 0) {
      focusableElements[0]?.focus()
    } else {
      // Если нет фокусируемых элементов, фокус на контейнере
      containerRef.value.setAttribute('tabindex', '-1')
      containerRef.value.focus()
    }
  }

  /**
   * Деактивирует focus trap и возвращает фокус
   */
  const deactivate = () => {
    isActive.value = false

    // Возвращаем фокус
    if (previousActiveElement.value && typeof previousActiveElement.value.focus === 'function') {
      previousActiveElement.value.focus()
    }

    previousActiveElement.value = null
  }

  /**
   * Обработчик Tab для удержания фокуса внутри контейнера
   */
  const handleTabKey = (event: KeyboardEvent) => {
    if (!isActive.value || event.key !== 'Tab') return

    const focusableElements = getFocusableElements()

    if (focusableElements.length === 0) {
      event.preventDefault()
      return
    }

    const first = focusableElements[0]
    const last = focusableElements[focusableElements.length - 1]

    // Shift + Tab на первом элементе → переход к последнему
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last?.focus()
      return
    }

    // Tab на последнем элементе → переход к первому
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first?.focus()
    }
  }

  /**
   * Обработчик Escape
   */
  const handleEscapeKey = (event: KeyboardEvent) => {
    if (!isActive.value || !escapeDeactivates || event.key !== 'Escape') return

    event.preventDefault()
    onEscape?.()
  }

  /**
   * Общий обработчик клавиатуры
   */
  const handleKeydown = (event: KeyboardEvent) => {
    handleTabKey(event)
    handleEscapeKey(event)
  }

  // Добавляем слушатели при монтировании
  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })

  // Удаляем слушатели при размонтировании
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
    if (isActive.value) {
      deactivate()
    }
  })

  return {
    activate,
    deactivate,
    isActive,
  }
}
