/**
 * useToast.ts - Enterprise Toast Notification System
 * Provides centralized user feedback with auto-dismiss and action support
 */

import { ref, readonly } from 'vue'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface ToastAction {
  label: string
  handler: () => void | Promise<void>
}

export interface Toast {
  id: string
  type: ToastType
  message: string
  description?: string
  duration?: number
  action?: ToastAction
  dismissible?: boolean
}

const toasts = ref<Toast[]>([])
let toastId = 0

export function useToast() {
  function show(
    type: ToastType,
    message: string,
    options: Partial<Omit<Toast, 'id' | 'type' | 'message'>> = {}
  ): string {
    const id = `toast-${++toastId}`
    const duration = options.duration ?? (type === 'error' ? 7000 : 4000)
    
    const toast: Toast = {
      id,
      type,
      message,
      description: options.description,
      duration,
      action: options.action,
      dismissible: options.dismissible ?? true
    }
    
    toasts.value.push(toast)
    
    if (duration > 0) {
      setTimeout(() => dismiss(id), duration)
    }
    
    return id
  }
  
  function success(message: string, description?: string) {
    return show('success', message, { description })
  }
  
  function error(message: string, description?: string, action?: ToastAction) {
    return show('error', message, { description, action })
  }
  
  function warning(message: string, description?: string) {
    return show('warning', message, { description })
  }
  
  function info(message: string, description?: string) {
    return show('info', message, { description })
  }
  
  function dismiss(id: string) {
    const index = toasts.value.findIndex(t => t.id === id)
    if (index !== -1) {
      toasts.value.splice(index, 1)
    }
  }
  
  function dismissAll() {
    toasts.value = []
  }
  
  return {
    toasts: readonly(toasts),
    show,
    success,
    error,
    warning,
    info,
    dismiss,
    dismissAll
  }
}