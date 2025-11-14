<template>
  <div class="w-full">
    <div
      ref="sliderRef"
      class="relative h-2 bg-steel-dark border border-steel-medium rounded-full cursor-pointer"
      @mousedown="startDrag"
      @touchstart="startDrag"
    >
      <div 
        class="absolute h-full rounded-full transition-all bg-gradient-to-r from-primary-600 to-primary-500" 
        :style="{ 
          width: fillPercentage + '%',
          boxShadow: '0 0 10px rgba(79, 70, 229, 0.3)'
        }" 
      />
      <div
        class="absolute w-4 h-4 bg-gradient-to-br from-steel-shine to-steel-light border-2 border-primary-500 rounded-full shadow-metal cursor-grab active:cursor-grabbing transform -translate-y-1/2 top-1/2 transition-all hover:scale-110"
        :style="{ left: `calc(${thumbPercentage}% - 8px)` }"
        @mousedown="startDrag"
        @touchstart="startDrag"
      />
    </div>
    <div v-if="showValue" class="flex justify-between text-sm text-text-secondary mt-2">
      <span>{{ props.min }}</span>
      <span class="font-medium text-text-primary">{{ modelValue }}</span>
      <span>{{ props.max }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Props {
  modelValue: number
  min?: number
  max?: number
  step?: number
  showValue?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  min: 0,
  max: 100,
  step: 1,
  showValue: true
})

const emit = defineEmits<{ 'update:modelValue': [value: number] }>()

const sliderRef = ref<HTMLElement>()
const isDragging = ref(false)

const thumbPercentage = computed(() => {
  const range = props.max - props.min
  return ((props.modelValue - props.min) / range) * 100
})
const fillPercentage = computed(() => thumbPercentage.value)

const getSteppedValue = (value: number): number => {
  const steps = Math.round((value - props.min) / props.step)
  return Math.min(props.max, Math.max(props.min, props.min + steps * props.step))
}

const updateValue = (clientX: number) => {
  if (!sliderRef.value) return
  const rect = sliderRef.value.getBoundingClientRect()
  const percentage = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
  const range = props.max - props.min
  const rawValue = props.min + percentage * range
  const steppedValue = getSteppedValue(rawValue)
  emit('update:modelValue', steppedValue)
}

const startDrag = (event: MouseEvent | TouchEvent) => {
  event.preventDefault()
  isDragging.value = true
  const touch = 'touches' in event ? (event.touches?.[0] ?? null) : null
  const clientX = touch ? touch.clientX : (event as MouseEvent).clientX
  updateValue(clientX)
}

const onMouseMove = (event: MouseEvent) => {
  if (!isDragging.value) return
  updateValue(event.clientX)
}

const onTouchMove = (event: TouchEvent) => {
  if (!isDragging.value) return
  event.preventDefault()
  const t0 = event.touches?.[0]
  if (!t0) return
  updateValue(t0.clientX)
}

const stopDrag = () => { isDragging.value = false }

const onKeyDown = (event: KeyboardEvent) => {
  if (!sliderRef.value?.contains(event.target as Node)) return
  let newValue = props.modelValue
  switch (event.key) {
    case 'ArrowRight':
    case 'ArrowUp':
      event.preventDefault(); newValue = Math.min(props.max, newValue + props.step); break
    case 'ArrowLeft':
    case 'ArrowDown':
      event.preventDefault(); newValue = Math.max(props.min, newValue - props.step); break
    case 'Home': event.preventDefault(); newValue = props.min; break
    case 'End': event.preventDefault(); newValue = props.max; break
    default: return
  }
  emit('update:modelValue', newValue)
}

onMounted(() => {
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', stopDrag)
  document.addEventListener('touchmove', onTouchMove, { passive: false })
  document.addEventListener('touchend', stopDrag)
  document.addEventListener('keydown', onKeyDown)
  sliderRef.value?.setAttribute('tabindex', '0')
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', stopDrag)
  document.removeEventListener('touchmove', onTouchMove)
  document.removeEventListener('touchend', stopDrag)
  document.removeEventListener('keydown', onKeyDown)
})
</script>