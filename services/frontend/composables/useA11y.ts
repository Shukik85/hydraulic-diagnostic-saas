/**
 * Composable для работы с accessibility
 * Предоставляет утилиты для announce messages, генерации ID и других a11y функций
 */

import { ref } from '#imports'
import type { Ref } from 'vue'

export const useA11y = () => {
  /**
   * Объявляет сообщение для screen readers через live regions
   * @param message - Текст сообщения для объявления
   * @param priority - Приоритет объявления ('polite' или 'assertive')
   */
  const announceMessage = (
    message: string,
    priority: 'polite' | 'assertive' = 'polite'
  ) => {
    const announcement = document.createElement('div')
    announcement.setAttribute('role', priority === 'assertive' ? 'alert' : 'status')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.className = 'sr-only'
    announcement.textContent = message

    document.body.appendChild(announcement)

    // Удаляем элемент после объявления
    setTimeout(() => {
      if (document.body.contains(announcement)) {
        document.body.removeChild(announcement)
      }
    }, 1000)
  }

  /**
   * Генерирует уникальный ID для связывания элементов accessibility
   * @param prefix - Префикс для ID (по умолчанию 'a11y')
   * @returns Уникальный ID строка
   */
  const generateId = (prefix = 'a11y') => {
    return `${prefix}-${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Объявляет об изменении статуса загрузки
   * @param isLoading - Статус загрузки
   * @param loadingMessage - Сообщение при загрузке
   * @param completeMessage - Сообщение при завершении
   */
  const announceLoadingStatus = (
    isLoading: boolean,
    loadingMessage: string,
    completeMessage: string
  ) => {
    if (isLoading) {
      announceMessage(loadingMessage, 'polite')
    } else {
      announceMessage(completeMessage, 'polite')
    }
  }

  /**
   * Объявляет об ошибке с высоким приоритетом
   * @param errorMessage - Текст сообщения об ошибке
   */
  const announceError = (errorMessage: string) => {
    announceMessage(errorMessage, 'assertive')
  }

  /**
   * Объявляет об успешном действии
   * @param successMessage - Текст сообщения об успехе
   */
  const announceSuccess = (successMessage: string) => {
    announceMessage(successMessage, 'polite')
  }

  /**
   * Проверяет наличие prefers-reduced-motion
   * @returns true если пользователь предпочитает уменьшенную анимацию
   */
  const prefersReducedMotion = () => {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  }

  return {
    announceMessage,
    generateId,
    announceLoadingStatus,
    announceError,
    announceSuccess,
    prefersReducedMotion,
  }
}
