<template>
  <div
    :class="cn(
      'relative flex w-full touch-none items-center select-none',
      orientation === 'vertical' ? 'h-full min-h-44 w-auto flex-col' : 'h-4',
      disabled && 'opacity-50',
      className
    )"
    v-bind="$attrs"
  >
    <div
      :class="cn(
        'relative grow overflow-hidden rounded-full bg-muted',
        orientation === 'vertical' ? 'h-full w-1.5' : 'h-full w-full'
      )"
    >
      <div
        :class="cn(
          'absolute bg-primary',
          orientation === 'vertical' ? 'w-full' : 'h-full'
        )"
        :style="rangeStyle"
      />
    </div>
    <div
      v-for="(thumb, index) in thumbs"
      :key="index"
      :class="cn(
        'absolute block size-4 shrink-0 rounded-full border border-primary bg-background shadow-sm transition-colors hover:ring-4 hover:ring-ring/50 focus-visible:ring-4 focus-visible:ring-ring/50 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50',
        orientation === 'vertical' ? 'left-1/2 -translate-x-1/2' : 'top-1/2 -translate-y-1/2'
      )"
      :style="thumbStyle(thumb)"
      @mousedown="startDrag(index, $event)"
      @touchstart="startDrag(index, $event)"
      tabindex="0"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { cn } from "./utils";

interface Props {
  modelValue?: number[];
  min?: number;
  max?: number;
  step?: number;
  orientation?: 'horizontal' | 'vertical';
  disabled?: boolean;
  className?: string;
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: () => [0],
  min: 0,
  max: 100,
  step: 1,
  orientation: 'horizontal',
  disabled: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: number[]];
}>();

const dragging = ref<number | null>(null);
const containerRef = ref<HTMLElement>();

const thumbs = computed(() => {
  return props.modelValue.map((value, index) => ({
    value,
    index,
    position: ((value - props.min) / (props.max - props.min)) * 100,
  }));
});

const rangeStyle = computed(() => {
  const minValue = Math.min(...props.modelValue);
  const maxValue = Math.max(...props.modelValue);
  const start = ((minValue - props.min) / (props.max - props.min)) * 100;
  const end = ((maxValue - props.min) / (props.max - props.min)) * 100;

  if (props.orientation === 'vertical') {
    return {
      bottom: `${start}%`,
      height: `${end - start}%`,
    };
  } else {
    return {
      left: `${start}%`,
      width: `${end - start}%`,
    };
  }
});

const thumbStyle = (thumb: any) => {
  if (props.orientation === 'vertical') {
    return {
      bottom: `${thumb.position}%`,
    };
  } else {
    return {
      left: `${thumb.position}%`,
    };
  }
};

const startDrag = (index: number, event: MouseEvent | TouchEvent) => {
  if (props.disabled) return;
  dragging.value = index;
  document.addEventListener('mousemove', onDrag);
  document.addEventListener('touchmove', onDrag);
  document.addEventListener('mouseup', stopDrag);
  document.addEventListener('touchend', stopDrag);
};

const onDrag = (event: MouseEvent | TouchEvent) => {
  if (dragging.value === null || !containerRef.value) return;

  const rect = containerRef.value.getBoundingClientRect();
  const clientX = 'touches' in event ? event.touches[0].clientX : event.clientX;
  const clientY = 'touches' in event ? event.touches[0].clientY : event.clientY;

  let percent;
  if (props.orientation === 'vertical') {
    percent = 1 - (clientY - rect.top) / rect.height;
  } else {
    percent = (clientX - rect.left) / rect.width;
  }

  percent = Math.max(0, Math.min(1, percent));
  const value = Math.round((percent * (props.max - props.min) + props.min) / props.step) * props.step;

  const newValues = [...props.modelValue];
  newValues[dragging.value] = value;
  emit('update:modelValue', newValues);
};

const stopDrag = () => {
  dragging.value = null;
  document.removeEventListener('mousemove', onDrag);
  document.removeEventListener('touchmove', onDrag);
  document.removeEventListener('mouseup', stopDrag);
  document.removeEventListener('touchend', stopDrag);
};

onMounted(() => {
  containerRef.value = document.querySelector('.slider-container') as HTMLElement;
});

onUnmounted(() => {
  stopDrag();
});
</script>
