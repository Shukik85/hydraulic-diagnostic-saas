/**
 * Toast notification composable
 */

import type { ToastMessage, ToastType } from '~/types';

export interface UseToastReturn {
  success: (message: string, title?: string, duration?: number) => void;
  error: (message: string, title?: string, duration?: number) => void;
  warning: (message: string, title?: string, duration?: number) => void;
  info: (message: string, title?: string, duration?: number) => void;
  dismiss: (id: string) => void;
  dismissAll: () => void;
}

/**
 * Toast notification composable
 */
export const useToast = (): UseToastReturn => {
  const uiStore = useUiStore();

  /**
   * Show toast with specific type
   */
  const showToast = (
    type: ToastType,
    message: string,
    title?: string,
    duration = 5000
  ): void => {
    const toast: ToastMessage = {
      id: `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      title,
      message,
      duration,
      dismissible: true,
    };

    uiStore.addToast(toast);

    // Auto-dismiss after duration
    if (duration > 0) {
      setTimeout(() => {
        uiStore.removeToast(toast.id);
      }, duration);
    }
  };

  return {
    success: (message: string, title?: string, duration?: number): void => {
      showToast('success', message, title, duration);
    },

    error: (message: string, title?: string, duration?: number): void => {
      showToast('error', message, title, duration ?? 0); // Errors don't auto-dismiss
    },

    warning: (message: string, title?: string, duration?: number): void => {
      showToast('warning', message, title, duration);
    },

    info: (message: string, title?: string, duration?: number): void => {
      showToast('info', message, title, duration);
    },

    dismiss: (id: string): void => {
      uiStore.removeToast(id);
    },

    dismissAll: (): void => {
      uiStore.clearToasts();
    },
  };
};
