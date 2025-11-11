/**
 * useErrorHandler.ts - Centralized Error Handling
 * Converts backend errors to user-friendly messages with actionable feedback
 */

import { useToast } from './useToast'

export interface ErrorContext {
  operation: string
  silent?: boolean
  retryable?: boolean
  onRetry?: () => void | Promise<void>
}

export interface ApiError {
  message: string
  code: string | number
  status?: number
  details?: unknown
}

const ERROR_MESSAGES: Record<number | string, string> = {
  // Client errors
  400: 'Неверный запрос. Проверьте введённые данные',
  401: 'Требуется авторизация. Войдите в систему',
  403: 'Доступ запрещён. Недостаточно прав',
  404: 'Ресурс не найден',
  409: 'Конфликт данных. Возможно, запись уже существует',
  422: 'Ошибка валидации данных',
  429: 'Слишком много запросов. Попробуйте позже',
  
  // Server errors
  500: 'Внутренняя ошибка сервера',
  502: 'Сервер недоступен',
  503: 'Сервис временно недоступен',
  504: 'Превышено время ожидания ответа',
  
  // Custom errors
  NETWORK_ERROR: 'Ошибка сети. Проверьте подключение к интернету',
  TIMEOUT: 'Превышено время ожидания',
  UNKNOWN: 'Неизвестная ошибка'
}

export function useErrorHandler() {
  const toast = useToast()
  
  function parseError(error: unknown): ApiError {
    // Fetch/Axios error
    if (error && typeof error === 'object') {
      const e = error as any
      
      if (e.response) {
        return {
          message: e.response.data?.message || e.response.data?.detail || ERROR_MESSAGES[e.response.status] || 'Unknown error',
          code: e.response.status,
          status: e.response.status,
          details: e.response.data
        }
      }
      
      if (e.data?.error) {
        return {
          message: e.data.error.message || ERROR_MESSAGES[e.data.error.code] || 'Unknown error',
          code: e.data.error.code || 'UNKNOWN',
          status: e.status
        }
      }
      
      if (e.message) {
        return {
          message: e.message,
          code: e.code || 'UNKNOWN'
        }
      }
    }
    
    return {
      message: String(error),
      code: 'UNKNOWN'
    }
  }
  
  function getUserFriendlyMessage(error: ApiError, context?: ErrorContext): string {
    const operation = context?.operation || 'Операция'
    const baseMessage = error.message || ERROR_MESSAGES[error.code] || ERROR_MESSAGES.UNKNOWN
    
    // Add context to generic errors
    if ([500, 502, 503, 504].includes(error.status as number)) {
      return `${operation} не выполнена: ${baseMessage}`
    }
    
    return baseMessage
  }
  
  function handle(error: unknown, context?: ErrorContext) {
    const parsedError = parseError(error)
    
    // Silent errors don't show toasts
    if (context?.silent) {
      console.error('[Error Handler]', parsedError)
      return parsedError
    }
    
    const message = getUserFriendlyMessage(parsedError, context)
    
    // Show toast with retry option if available
    if (context?.retryable && context.onRetry) {
      toast.error(message, undefined, {
        label: 'Повторить',
        handler: context.onRetry
      })
    } else {
      toast.error(message)
    }
    
    console.error('[Error Handler]', parsedError)
    return parsedError
  }
  
  function handleApiError(
    error: unknown,
    operation: string,
    options?: Omit<ErrorContext, 'operation'>
  ) {
    return handle(error, { operation, ...options })
  }
  
  return {
    handle,
    handleApiError,
    parseError
  }
}