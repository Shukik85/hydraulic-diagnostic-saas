/**
 * Toast Composable
 * Toast notifications with auto-dismiss
 */

import type { ToastMessage, ToastType } from '~/types';

const toasts = ref<ToastMessage[]>([]);
let idCounter = 0;

const DEFAULT_DURATION = 5000;

/**
 * Generate unique toast ID
 */
function generateId(): string {
  return `toast-${Date.now()}-${idCounter++}`;
}

/**
 * Remove toast after duration
 */
function scheduleRemoval(id: string, duration: number): void {
  setTimeout(() => {
    remove(id);
  }, duration);
}

export function useToast() {
  /**
   * Add toast notification
   */
  function add(
    message: string,
    type: ToastType = 'info',
    title?: string,
    duration: number = DEFAULT_DURATION
  ): string {
    const id = generateId();

    const toast: ToastMessage = {
      id,
      type,
      title,
      message,
      duration,
      dismissible: true,
      createdAt: new Date(),
    };

    toasts.value.push(toast);

    if (duration > 0) {
      scheduleRemoval(id, duration);
    }

    return id;
  }

  /**
   * Remove toast by ID
   */
  function remove(id: string): void {
    const index = toasts.value.findIndex((t) => t.id === id);
    if (index !== -1) {
      toasts.value.splice(index, 1);
    }
  }

  /**
   * Clear all toasts
   */
  function clear(): void {
    toasts.value = [];
  }

  /**
   * Success toast
   */
  function success(message: string, title: string = 'Success'): string {
    return add(message, 'success', title);
  }

  /**
   * Error toast
   */
  function error(message: string, title: string = 'Error'): string {
    return add(message, 'error', title, 0); // Don't auto-dismiss errors
  }

  /**
   * Warning toast
   */
  function warning(message: string, title: string = 'Warning'): string {
    return add(message, 'warning', title);
  }

  /**
   * Info toast
   */
  function info(message: string, title: string = 'Info'): string {
    return add(message, 'info', title);
  }

  return {
    toasts: readonly(toasts),
    add,
    remove,
    clear,
    success,
    error,
    warning,
    info,
  };
}
