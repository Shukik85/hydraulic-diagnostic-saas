<template>
  <span :class="badgeClasses">
    <span v-if="withDot" :class="dotClasses"></span>
    <slot>{{ label }}</slot>
  </span>
</template>

<script setup lang="ts">
type StatusType = 'operational' | 'degraded' | 'critical' | 'info' | 'unknown'

interface Props {
  status: StatusType
  label?: string
  withDot?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  withDot: true
})

const statusConfig: Record<StatusType, { bg: string, text: string, dot: string }> = {
  operational: {
    bg: 'bg-green-100 dark:bg-green-900/30',
    text: 'text-green-800 dark:text-green-300',
    dot: 'bg-green-500'
  },
  degraded: {
    bg: 'bg-orange-100 dark:bg-orange-900/30',
    text: 'text-orange-800 dark:text-orange-300',
    dot: 'bg-orange-500'
  },
  critical: {
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-800 dark:text-red-300',
    dot: 'bg-red-500'
  },
  info: {
    bg: 'bg-cyan-100 dark:bg-cyan-900/30',
    text: 'text-cyan-800 dark:text-cyan-300',
    dot: 'bg-cyan-500'
  },
  unknown: {
    bg: 'bg-gray-100 dark:bg-gray-800',
    text: 'text-gray-600 dark:text-gray-400',
    dot: 'bg-gray-500'
  }
}

const sizeClasses = {
  sm: 'text-xs px-2 py-0.5 gap-1',
  md: 'text-sm px-2.5 py-1 gap-1.5',
  lg: 'text-base px-3 py-1.5 gap-2'
}

const dotSizes = {
  sm: 'w-1.5 h-1.5',
  md: 'w-2 h-2',
  lg: 'w-2.5 h-2.5'
}

const badgeClasses = computed(() => [
  'inline-flex items-center font-medium rounded-full',
  statusConfig[props.status].bg,
  statusConfig[props.status].text,
  sizeClasses[props.size]
])

const dotClasses = computed(() => [
  'rounded-full',
  statusConfig[props.status].dot,
  dotSizes[props.size]
])
</script>