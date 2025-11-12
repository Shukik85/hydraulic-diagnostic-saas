/**
 * useDebounceThrottle.ts — Debounce и Throttle utilities
 * 
 * Оптимизирует обработку частых событий (ввод, scroll, resize)
 */
import { ref, watch, onUnmounted, customRef, type Ref } from 'vue'

/**
 * Debounced ref - задержка обновления значения
 */
export function useDebouncedRef<T>(initialValue: T, delay: number) {
  const immediate = ref<T>(initialValue)
  const debounced = ref<T>(initialValue)
  
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  watch(immediate, (newValue) => {
    if (timeoutId) clearTimeout(timeoutId)
    
    timeoutId = setTimeout(() => {
      debounced.value = newValue
    }, delay)
  })
  
  onUnmounted(() => {
    if (timeoutId) clearTimeout(timeoutId)
  })
  
  return { immediate, debounced }
}

/**
 * Custom debounced ref using customRef
 */
export function debouncedRef<T>(value: T, delay = 300): Ref<T> {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  return customRef((track, trigger) => {
    return {
      get() {
        track()
        return value
      },
      set(newValue: T) {
        if (timeoutId) clearTimeout(timeoutId)
        
        timeoutId = setTimeout(() => {
          value = newValue
          trigger()
        }, delay)
      }
    }
  })
}

/**
 * Debounce function
 */
export function useDebounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  const debouncedFn = (...args: Parameters<T>) => {
    if (timeoutId) clearTimeout(timeoutId)
    
    timeoutId = setTimeout(() => {
      fn(...args)
    }, delay)
  }
  
  onUnmounted(() => {
    if (timeoutId) clearTimeout(timeoutId)
  })
  
  return debouncedFn
}

/**
 * Throttle function - ограничение частоты вызова
 */
export function useThrottle<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  const throttledFn = (...args: Parameters<T>) => {
    const now = Date.now()
    const timeSinceLastCall = now - lastCall
    
    if (timeSinceLastCall >= delay) {
      lastCall = now
      fn(...args)
    } else {
      // Schedule call for remaining time
      if (timeoutId) clearTimeout(timeoutId)
      
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        fn(...args)
      }, delay - timeSinceLastCall)
    }
  }
  
  onUnmounted(() => {
    if (timeoutId) clearTimeout(timeoutId)
  })
  
  return throttledFn
}

/**
 * Throttled ref
 */
export function throttledRef<T>(value: T, delay = 300): Ref<T> {
  let lastUpdate = 0
  let pendingValue: T | null = null
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  return customRef((track, trigger) => {
    return {
      get() {
        track()
        return value
      },
      set(newValue: T) {
        const now = Date.now()
        const timeSinceLastUpdate = now - lastUpdate
        
        if (timeSinceLastUpdate >= delay) {
          value = newValue
          lastUpdate = now
          trigger()
        } else {
          pendingValue = newValue
          
          if (timeoutId) clearTimeout(timeoutId)
          
          timeoutId = setTimeout(() => {
            if (pendingValue !== null) {
              value = pendingValue
              pendingValue = null
              lastUpdate = Date.now()
              trigger()
            }
          }, delay - timeSinceLastUpdate)
        }
      }
    }
  })
}

/**
 * UseDebounce for watch
 */
export function useDebouncedWatch<T>(
  source: Ref<T>,
  callback: (value: T, oldValue: T) => void,
  delay: number
) {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  const stopWatch = watch(source, (newValue, oldValue) => {
    if (timeoutId) clearTimeout(timeoutId)
    
    timeoutId = setTimeout(() => {
      callback(newValue, oldValue)
    }, delay)
  })
  
  onUnmounted(() => {
    if (timeoutId) clearTimeout(timeoutId)
    stopWatch()
  })
  
  return stopWatch
}

/**
 * UseThrottle for watch
 */
export function useThrottledWatch<T>(
  source: Ref<T>,
  callback: (value: T, oldValue: T) => void,
  delay: number
) {
  let lastCall = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  const stopWatch = watch(source, (newValue, oldValue) => {
    const now = Date.now()
    const timeSinceLastCall = now - lastCall
    
    if (timeSinceLastCall >= delay) {
      lastCall = now
      callback(newValue, oldValue)
    } else {
      if (timeoutId) clearTimeout(timeoutId)
      
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        callback(newValue, oldValue)
      }, delay - timeSinceLastCall)
    }
  })
  
  onUnmounted(() => {
    if (timeoutId) clearTimeout(timeoutId)
    stopWatch()
  })
  
  return stopWatch
}

/**
 * Debounced search composable
 */
export function useDebouncedSearch(
  searchFn: (query: string) => Promise<any>,
  delay = 300
) {
  const query = ref('')
  const results = ref<any[]>([])
  const isSearching = ref(false)
  const error = ref<string | null>(null)
  
  const debouncedSearch = useDebounce(async (q: string) => {
    if (!q.trim()) {
      results.value = []
      return
    }
    
    isSearching.value = true
    error.value = null
    
    try {
      results.value = await searchFn(q)
    } catch (err: any) {
      error.value = err.message || 'Search failed'
      results.value = []
    } finally {
      isSearching.value = false
    }
  }, delay)
  
  watch(query, (newQuery) => {
    debouncedSearch(newQuery)
  })
  
  return {
    query,
    results,
    isSearching,
    error
  }
}
