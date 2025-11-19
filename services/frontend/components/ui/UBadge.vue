<template>
  <component
    :is="props.asChild ? 'slot' : 'span'"
    data-slot="badge"
    :class="cn(badgeVariants({ variant: props.variant }), props.className)"
    v-bind="$attrs"
  >
    <slot />
  </component>
</template>
<script setup lang="ts">
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from './utils'
const badgeVariants = cva('badge-status transition-[color,box-shadow] overflow-hidden', {
  variants: {
    variant: {
      default: 'bg-primary-500/20 text-primary-300 border-primary-500/30',
      success: 'badge-success',
      warning: 'badge-warning',
      error: 'badge-error',
      info: 'badge-info',
      secondary: 'bg-steel-dark/20 text-steel-shine border-steel-medium/30',
      destructive: 'badge-error',
      outline: 'bg-transparent text-text-primary border-steel-medium [a&]:hover:bg-background-hover',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})
interface Props {
  variant?: VariantProps<typeof badgeVariants>["variant"];
  asChild?: boolean;
  className?: string;
}
const props = withDefaults(defineProps<Props>(), { variant: 'default', asChild: false })
</script>