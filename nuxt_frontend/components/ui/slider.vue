<template>
  <div
    ref="containerRef"
    :class="[
      'relative flex items-center select-none touch-none',
      props.orientation === 'vertical' ? 'flex-col h-full' : 'w-full',
      props.disabled && 'opacity-50 cursor-not-allowed'
    ]"
    @mousedown="!props.disabled && handlePointerDown"
    @touchstart="!props.disabled && handlePointerDown"
  >
    <!-- Track -->
    <div
      :class="[
        'relative rounded-full bg-gray-200 dark:bg-gray-700',
        props.orientation === 'vertical' ? 'w-2 flex-1' : 'h-2 w-full'
      ]"
    >
      <!-- Range/Progress -->
      <div
        :class="[
          'absolute rounded-full bg-blue-600 transition-all duration-150',
          props.orientation === 'vertical' ? 'w-full' : 'h-full'
        ]"
        :style="rangeStyle"
      />
    </div>
    
    <!-- Thumbs -->
    <div
      v-for="(value, index) in normalizedValues"
      :key="index"
      :class="[
        'absolute w-4 h-4 bg-white border-2 border-blue-600 rounded-full shadow-md transition-all duration-150',
        'hover:scale-110 focus:scale-110 focus:outline-none focus:ring-2 focus:ring-blue-500',
        props.disabled ? 'cursor-not-allowed' : 'cursor-grab active:cursor-grabbing'
      ]"
      :style="getThumbStyle(value)"
      :tabindex="props.disabled ? -1 : 0"
      @keydown="(e) => handleKeyDown(e, index)"
    />
  </div>
</template>

<script setup lang="ts">
interface SliderProps {
  modelValue?: number | number[]
  min?: number
  max?: number
  step?: number
  orientation?: 'horizontal' | 'vertical'
  disabled?: boolean
  class?: string
}

interface SliderEmits {
  'update:modelValue': [value: number | number[]]
}

const props = withDefaults(defineProps<SliderProps>(), {
  modelValue: 0,
  min: 0,
  max: 100,
  step: 1,
  orientation: 'horizontal',
  disabled: false
})

const emit = defineEmits<SliderEmits>()

const containerRef = ref<HTMLElement>()

// Normalize value to array for consistent handling
const normalizedValues = computed(() => {
  const value = props.modelValue
  return Array.isArray(value) ? value : [value]
})

// Calculate range style for visual feedback
const rangeStyle = computed(() => {
  const values = normalizedValues.value
  const min = Math.min(...values)
  const max = Math.max(...values)
  
  const minPercent = ((min - props.min) / (props.max - props.min)) * 100
  const maxPercent = ((max - props.min) / (props.max - props.min)) * 100
  
  if (props.orientation === 'vertical') {
    return {
      bottom: `${minPercent}%`,
      height: `${maxPercent - minPercent}%`
    }
  } else {
    return {
      left: `${minPercent}%`,
      width: `${maxPercent - minPercent}%`
    }
  }
})

// Calculate thumb position
const getThumbStyle = (value: number) => {
  const percent = ((value - props.min) / (props.max - props.min)) * 100
  
  if (props.orientation === 'vertical') {
    return {
      bottom: `${percent}%`,
      left: '50%',
      transform: 'translateX(-50%)'
    }
  } else {
    return {
      left: `${percent}%`,
      top: '50%',
      transform: 'translateY(-50%)'
    }
  }
}

// Handle pointer events with proper type safety
const handlePointerDown = (event: MouseEvent | TouchEvent) => {
  if (props.disabled || !containerRef.value) return
  
  event.preventDefault()
  
  const rect = containerRef.value.getBoundingClientRect()
  
  // Fixed: Proper type checking for touch events
  let clientX: number
  let clientY: number
  
  if ('touches' in event && event.touches && event.touches[0]) {
    clientX = event.touches[0].clientX
    clientY = event.touches[0].clientY
  } else if ('clientX' in event) {
    clientX = (event as MouseEvent).clientX
    clientY = (event as MouseEvent).clientY
  } else {
    return // Can't determine coordinates
  }
  
  let percent: number
  if (props.orientation === 'vertical') {
    percent = ((rect.bottom - clientY) / rect.height) * 100
  } else {
    percent = ((clientX - rect.left) / rect.width) * 100
  }
  
  // Clamp percentage
  percent = Math.max(0, Math.min(100, percent))
  
  // Calculate new value
  const newValue = props.min + (percent / 100) * (props.max - props.min)
  const steppedValue = Math.round(newValue / props.step) * props.step
  
  // Emit new value
  if (Array.isArray(props.modelValue)) {
    // For range sliders, update closest thumb
    const values = [...props.modelValue]
    const closestIndex = values.reduce((closest, val, idx) => 
      Math.abs(val - steppedValue) < Math.abs(values[closest] - steppedValue) ? idx : closest, 0
    )
    values[closestIndex] = steppedValue
    emit('update:modelValue', values)
  } else {
    emit('update:modelValue', steppedValue)
  }
}

// Keyboard navigation
const handleKeyDown = (event: KeyboardEvent, thumbIndex: number) => {
  if (props.disabled) return
  
  const { key } = event
  const values = normalizedValues.value
  
  let newValue = values[thumbIndex]
  
  switch (key) {
    case 'ArrowRight':
    case 'ArrowUp':
      newValue = Math.min(props.max, newValue + props.step)
      event.preventDefault()
      break
    case 'ArrowLeft':
    case 'ArrowDown':
      newValue = Math.max(props.min, newValue - props.step)
      event.preventDefault()
      break
    case 'Home':
      newValue = props.min
      event.preventDefault()
      break
    case 'End':
      newValue = props.max
      event.preventDefault()
      break
    default:
      return
  }
  
  // Emit updated value
  if (Array.isArray(props.modelValue)) {
    const updatedValues = [...values]
    updatedValues[thumbIndex] = newValue
    emit('update:modelValue', updatedValues)
  } else {
    emit('update:modelValue', newValue)
  }
}
</script>