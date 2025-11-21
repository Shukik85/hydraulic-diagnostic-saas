/**
 * useKeyboardNav.ts - Keyboard navigation и focus management utilities
 * Обеспечивает accessibility для keyboard-only пользователей
 */

import { ref, onMounted, onUnmounted, type Ref } from '#imports'

/**
 * Опции для keyboard navigation
 */
export interface UseKeyboardNavOptions {
  onEscape?: () => void
  onEnter?: () => void
  onTab?: (event: KeyboardEvent) => void
  onArrowUp?: () => void
  onArrowDown?: () => void
  onArrowLeft?: () => void
  onArrowRight?: () => void
}

/**
 * Composable для обработки keyboard events
 * 
 * @example
 * ```typescript
 * const { handleKeydown } = useKeyboardNav({
 *   onEscape: () => closeModal(),
 *   onEnter: () => submitForm(),
 * })
 * ```
 */
export const useKeyboardNav = (options: UseKeyboardNavOptions = {}) => {
  const handleKeydown = (event: KeyboardEvent): void => {
    switch (event.key) {
      case 'Escape':
        options.onEscape?.()
        break
      case 'Enter':
        options.onEnter?.()
        break
      case 'Tab':
        options.onTab?.(event)
        break
      case 'ArrowUp':
        options.onArrowUp?.()
        break
      case 'ArrowDown':
        options.onArrowDown?.()
        break
      case 'ArrowLeft':
        options.onArrowLeft?.()
        break
      case 'ArrowRight':
        options.onArrowRight?.()
        break
    }
  }
  
  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })
  
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
  })
  
  return { handleKeydown }
}

/**
 * Получить все focusable элементы в контейнере
 */
const getFocusableElements = (container: HTMLElement): HTMLElement[] => {
  const selector = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',')
  
  return Array.from(
    container.querySelectorAll<HTMLElement>(selector)
  )
}

/**
 * Focus trap для модальных окон и диалогов
 * Предотвращает выход фокуса за пределы контейнера
 * 
 * @example
 * ```vue
 * <script setup lang="ts">
 * const modalRef = ref<HTMLElement | null>(null)
 * const { activate, deactivate } = useFocusTrap(modalRef)
 * 
 * const openModal = () => {
 *   activate()
 * }
 * 
 * const closeModal = () => {
 *   deactivate()
 * }
 * </script>
 * 
 * <template>
 *   <div ref="modalRef" role="dialog" aria-modal="true">
 *     <!-- Modal content -->
 *   </div>
 * </template>
 * ```
 */
export const useFocusTrap = (containerRef: Ref<HTMLElement | null>) => {
  const previousActiveElement = ref<HTMLElement | null>(null)
  
  const handleTabKey = (event: KeyboardEvent): void => {
    if (event.key !== 'Tab' || !containerRef.value) {
      return
    }
    
    const focusableElements = getFocusableElements(containerRef.value)
    
    if (focusableElements.length === 0) {
      return
    }
    
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]
    
    // Shift + Tab на первом элементе -> переход к последнему
    if (event.shiftKey && document.activeElement === firstElement) {
      event.preventDefault()
      if (lastElement) {
        lastElement.focus()
      }
    } 
    // Tab на последнем элементе -> переход к первому
    else if (!event.shiftKey && document.activeElement === lastElement) {
      event.preventDefault()
      if (firstElement) {
        firstElement.focus()
      }
    }
  }
  
  /**
   * Активировать focus trap
   */
  const activate = (): void => {
    if (!containerRef.value) {
      return
    }
    
    // Сохраняем текущий фокус
    previousActiveElement.value = document.activeElement as HTMLElement
    
    // Фокусируем первый focusable элемент
    const focusableElements = getFocusableElements(containerRef.value)
    if (focusableElements.length > 0 && focusableElements[0]) {
      focusableElements[0].focus()
    }
    
    // Подписываемся на Tab
    window.addEventListener('keydown', handleTabKey)
  }
  
  /**
   * Деактивировать focus trap
   */
  const deactivate = (): void => {
    // Возвращаем фокус
    if (previousActiveElement.value) {
      previousActiveElement.value.focus()
      previousActiveElement.value = null
    }
    
    // Отписываемся от Tab
    window.removeEventListener('keydown', handleTabKey)
  }
  
  // Cleanup при unmount
  onUnmounted(() => {
    deactivate()
  })
  
  return {
    activate,
    deactivate,
    getFocusableElements: () => 
      containerRef.value ? getFocusableElements(containerRef.value) : []
  }
}

/**
 * Автоматическое управление focus для roving tabindex
 * Используется для навигации по списку элементов стрелками
 * 
 * @example
 * ```vue
 * <script setup lang="ts">
 * const items = ref(['Item 1', 'Item 2', 'Item 3'])
 * const { currentIndex, focusNext, focusPrevious } = useRovingTabindex(items.value.length)
 * </script>
 * 
 * <template>
 *   <ul @keydown.arrow-down.prevent="focusNext" @keydown.arrow-up.prevent="focusPrevious">
 *     <li 
 *       v-for="(item, index) in items" 
 *       :key="index"
 *       :tabindex="currentIndex === index ? 0 : -1"
 *     >
 *       {{ item }}
 *     </li>
 *   </ul>
 * </template>
 * ```
 */
export const useRovingTabindex = (itemCount: number) => {
  const currentIndex = ref(0)
  
  const focusNext = (): void => {
    currentIndex.value = (currentIndex.value + 1) % itemCount
  }
  
  const focusPrevious = (): void => {
    currentIndex.value = currentIndex.value === 0 
      ? itemCount - 1 
      : currentIndex.value - 1
  }
  
  const focusFirst = (): void => {
    currentIndex.value = 0
  }
  
  const focusLast = (): void => {
    currentIndex.value = itemCount - 1
  }
  
  const setFocus = (index: number): void => {
    if (index >= 0 && index < itemCount) {
      currentIndex.value = index
    }
  }
  
  return {
    currentIndex,
    focusNext,
    focusPrevious,
    focusFirst,
    focusLast,
    setFocus
  }
}
