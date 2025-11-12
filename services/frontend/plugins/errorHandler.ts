/**
 * errorHandler.ts — Глобальный error handler
 * 
 * Features:
 * - Vue error handler
 * - Unhandled promise rejection handler
 * - Global error logger
 * - Toast notifications for errors
 * - Integration with Sentry (optional)
 */
export default defineNuxtPlugin((nuxtApp) => {
  const toast = useToast()
  const config = useRuntimeConfig()
  
  // Error statistics
  const errorStats = {
    vueErrors: 0,
    promiseRejections: 0,
    networkErrors: 0,
    lastError: null as Error | null,
    lastErrorTime: 0
  }
  
  /**
   * Vue error handler
   */
  nuxtApp.vueApp.config.errorHandler = (error: any, instance: any, info: string) => {
    console.error('[Vue Error]', error, info)
    errorStats.vueErrors++
    errorStats.lastError = error
    errorStats.lastErrorTime = Date.now()
    
    // Show user-friendly error message
    toast.add({
      title: 'Ошибка приложения',
      description: getErrorMessage(error),
      color: 'red',
      timeout: 5000,
      actions: [{
        label: 'Перезагрузить',
        click: () => window.location.reload()
      }]
    })
    
    // Send to monitoring service
    if (config.public.sentryDsn) {
      // Sentry.captureException(error, {
      //   contexts: {
      //     vue: {
      //       componentName: instance?.$options?.name,
      //       propsData: instance?.$props,
      //       lifecycle: info
      //     }
      //   }
      // })
    }
  }
  
  /**
   * Unhandled promise rejection handler
   */
  if (process.client) {
    window.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
      console.error('[Unhandled Promise Rejection]', event.reason)
      errorStats.promiseRejections++
      errorStats.lastError = event.reason
      errorStats.lastErrorTime = Date.now()
      
      // Don't show toast for network errors (handled by API client)
      if (event.reason?.code !== 'NETWORK_ERROR') {
        toast.add({
          title: 'Неожиданная ошибка',
          description: getErrorMessage(event.reason),
          color: 'red',
          timeout: 5000
        })
      }
      
      if (config.public.sentryDsn) {
        // Sentry.captureException(event.reason)
      }
    })
    
    /**
     * Global error handler for uncaught exceptions
     */
    window.addEventListener('error', (event: ErrorEvent) => {
      console.error('[Global Error]', event.error)
      
      // Don't show toast for script loading errors
      if (!event.filename.includes('chunk')) {
        toast.add({
          title: 'Критическая ошибка',
          description: event.message,
          color: 'red',
          timeout: 0 // Stay visible
        })
      }
      
      if (config.public.sentryDsn) {
        // Sentry.captureException(event.error)
      }
    })
  }
  
  /**
   * Get user-friendly error message
   */
  function getErrorMessage(error: any): string {
    if (typeof error === 'string') return error
    
    if (error?.message) {
      // Translate common error messages
      const translations: Record<string, string> = {
        'Network Error': 'Ошибка сети. Проверьте подключение',
        'Request timeout': 'Время ожидания истекло',
        'Access denied': 'Доступ запрещён',
        'Not found': 'Ресурс не найден',
        'Server error': 'Ошибка сервера'
      }
      
      return translations[error.message] || error.message
    }
    
    return 'Произошла ошибка'
  }
  
  /**
   * Provide global error logger
   */
  return {
    provide: {
      logError: (error: Error, context?: string, extra?: Record<string, any>) => {
        console.error(`[${context || 'App'}]`, error, extra)
        
        // Send to monitoring service in production
        if (process.env.NODE_ENV === 'production' && config.public.sentryDsn) {
          // Sentry.captureException(error, {
          //   tags: { context },
          //   extra
          // })
        }
      },
      
      logWarning: (message: string, context?: string, extra?: Record<string, any>) => {
        console.warn(`[${context || 'App'}]`, message, extra)
        
        if (process.env.NODE_ENV === 'production' && config.public.sentryDsn) {
          // Sentry.captureMessage(message, {
          //   level: 'warning',
          //   tags: { context },
          //   extra
          // })
        }
      },
      
      getErrorStats: () => errorStats
    }
  }
})

/**
 * TypeScript augmentation
 */
declare module '#app' {
  interface NuxtApp {
    $logError: (error: Error, context?: string, extra?: Record<string, any>) => void
    $logWarning: (message: string, context?: string, extra?: Record<string, any>) => void
    $getErrorStats: () => typeof errorStats
  }
}

declare module 'vue' {
  interface ComponentCustomProperties {
    $logError: (error: Error, context?: string, extra?: Record<string, any>) => void
    $logWarning: (message: string, context?: string, extra?: Record<string, any>) => void
    $getErrorStats: () => any
  }
}
