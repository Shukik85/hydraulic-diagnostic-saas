<template>
  <div :class="cardClasses">
    <div v-if="$slots.header || title" class="card-header">
      <slot name="header">
        <h3 v-if="title" class="card-title">{{ title }}</h3>
      </slot>
      <div v-if="$slots.actions" class="card-actions">
        <slot name="actions"></slot>
      </div>
    </div>
    
    <div :class="bodyClasses">
      <slot></slot>
    </div>
    
    <div v-if="$slots.footer" class="card-footer">
      <slot name="footer"></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
  bordered?: boolean
  elevated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  padding: 'md',
  hover: false,
  bordered: true,
  elevated: false
})

const cardClasses = computed(() => [
  'base-card',
  'bg-white dark:bg-industrial-800',
  'rounded-card',
  'transition-all duration-250',
  {
    'border border-industrial-200 dark:border-industrial-700': props.bordered,
    'shadow-card': !props.elevated,
    'shadow-elevated': props.elevated,
    'hover:shadow-card-hover hover:-translate-y-0.5': props.hover
  }
])

const bodyClasses = computed(() => [
  'card-body',
  {
    'p-0': props.padding === 'none',
    'p-3': props.padding === 'sm',
    'p-4': props.padding === 'md',
    'p-6': props.padding === 'lg'
  }
])
</script>

<style scoped>
.card-header {
  @apply flex items-center justify-between px-4 py-3 border-b border-industrial-200 dark:border-industrial-700;
}

.card-title {
  @apply text-lg font-semibold text-industrial-900 dark:text-industrial-50;
}

.card-actions {
  @apply flex items-center gap-2;
}

.card-footer {
  @apply px-4 py-3 border-t border-industrial-200 dark:border-industrial-700 bg-industrial-50 dark:bg-industrial-900/50;
}
</style>