/**
 * useVirtualScroll.ts — Virtual scrolling для больших списков
 * 
 * Оптимизирует рендеринг списков 1000+ элементов
 * Рендерит только видимые элементы + buffer
 */
import { ref, computed, type Ref } from 'vue'

export interface VirtualScrollOptions {
  itemHeight: number
  bufferSize?: number
  containerHeight?: number
}

export interface VirtualScrollItem<T> {
  item: T
  index: number
  top: number
}

/**
 * Virtual scroll composable
 */
export function useVirtualScroll<T>(
  items: Ref<T[]>,
  options: VirtualScrollOptions
) {
  const { itemHeight, bufferSize = 5, containerHeight = 600 } = options
  
  const scrollTop = ref(0)
  
  const visibleRange = computed(() => {
    const start = Math.max(0, Math.floor(scrollTop.value / itemHeight) - bufferSize)
    const visibleCount = Math.ceil(containerHeight / itemHeight)
    const end = Math.min(items.value.length, start + visibleCount + bufferSize * 2)
    
    return { start, end }
  })
  
  const visibleItems = computed((): VirtualScrollItem<T>[] => {
    const { start, end } = visibleRange.value
    return items.value.slice(start, end).map((item, idx) => ({
      item,
      index: start + idx,
      top: (start + idx) * itemHeight
    }))
  })
  
  const totalHeight = computed(() => items.value.length * itemHeight)
  
  const onScroll = (event: Event) => {
    scrollTop.value = (event.target as HTMLElement).scrollTop
  }
  
  const scrollToIndex = (index: number, behavior: ScrollBehavior = 'smooth') => {
    const container = document.querySelector('.virtual-scroll-container') as HTMLElement
    if (container) {
      container.scrollTo({
        top: index * itemHeight,
        behavior
      })
    }
  }
  
  return {
    visibleItems,
    totalHeight,
    onScroll,
    visibleRange,
    scrollToIndex,
    scrollTop: computed(() => scrollTop.value)
  }
}

/**
 * Variable height virtual scroll (more complex)
 */
export function useVariableHeightVirtualScroll<T>(
  items: Ref<T[]>,
  getItemHeight: (item: T, index: number) => number,
  options: { bufferSize?: number; containerHeight?: number } = {}
) {
  const { bufferSize = 5, containerHeight = 600 } = options
  
  const scrollTop = ref(0)
  const heightCache = new Map<number, number>()
  
  // Calculate cumulative heights
  const cumulativeHeights = computed(() => {
    const heights: number[] = [0]
    for (let i = 0; i < items.value.length; i++) {
      const cachedHeight = heightCache.get(i)
      const height = cachedHeight ?? getItemHeight(items.value[i], i)
      if (!cachedHeight) heightCache.set(i, height)
      heights.push(heights[i] + height)
    }
    return heights
  })
  
  const totalHeight = computed(() => {
    const heights = cumulativeHeights.value
    return heights[heights.length - 1] || 0
  })
  
  const visibleRange = computed(() => {
    const heights = cumulativeHeights.value
    
    // Binary search for start index
    let start = 0
    let end = items.value.length
    while (start < end) {
      const mid = Math.floor((start + end) / 2)
      if (heights[mid] < scrollTop.value) {
        start = mid + 1
      } else {
        end = mid
      }
    }
    start = Math.max(0, start - bufferSize)
    
    // Find end index
    const viewportEnd = scrollTop.value + containerHeight
    end = start
    while (end < items.value.length && heights[end] < viewportEnd) {
      end++
    }
    end = Math.min(items.value.length, end + bufferSize)
    
    return { start, end }
  })
  
  const visibleItems = computed(() => {
    const { start, end } = visibleRange.value
    const heights = cumulativeHeights.value
    
    return items.value.slice(start, end).map((item, idx) => ({
      item,
      index: start + idx,
      top: heights[start + idx],
      height: heightCache.get(start + idx) || 0
    }))
  })
  
  const onScroll = (event: Event) => {
    scrollTop.value = (event.target as HTMLElement).scrollTop
  }
  
  const updateItemHeight = (index: number, height: number) => {
    heightCache.set(index, height)
  }
  
  return {
    visibleItems,
    totalHeight,
    onScroll,
    visibleRange,
    updateItemHeight,
    scrollTop: computed(() => scrollTop.value)
  }
}
