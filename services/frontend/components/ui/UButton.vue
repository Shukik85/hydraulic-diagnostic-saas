<script setup lang="ts">
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from './utils'

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-5 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-primary-500/20 focus-visible:ring-[3px]",
  {
    variants: {
      variant: {
        default: 'bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-800 shadow-sm',
        destructive: 'bg-red-600 text-white hover:bg-red-700 active:bg-red-800 shadow-sm',
        outline: 'border-2 border-steel-600 bg-transparent text-white hover:bg-steel-800/50 active:bg-steel-800',
        secondary: 'bg-steel-700 text-white hover:bg-steel-600 active:bg-steel-500 shadow-sm',
        ghost: 'hover:bg-steel-800/50 active:bg-steel-800 text-white',
        link: 'text-primary-400 underline-offset-4 hover:underline hover:text-primary-300',
      },
      size: {
        sm: 'h-10 px-4 text-sm rounded-md',  // 40px - минимум для второстепенных действий
        default: 'h-12 px-6 text-base',      // 48px - стандарт (touch-friendly)
        lg: 'h-14 px-8 text-lg',             // 56px - для главных CTA
        xl: 'h-16 px-10 text-xl',            // 64px - для hero sections
        icon: 'size-12 rounded-lg',          // 48x48 - иконочные кнопки
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

interface Props {
  variant?: VariantProps<typeof buttonVariants>['variant']
  size?: VariantProps<typeof buttonVariants>['size']
  asChild?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
  asChild: false
})
</script>

<template>
  <component
    :is="props.asChild ? 'slot' : 'button'"
    :class="cn(buttonVariants({ variant: props.variant, size: props.size, className: props.className }))"
    data-slot="button"
    v-bind="$attrs"
  >
    <slot />
  </component>
</template>
