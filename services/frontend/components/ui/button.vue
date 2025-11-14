<template>
  <component
    :is="asChild ? 'slot' : 'button'"
    :class="cn(buttonVariants({ variant, size, className }))"
    data-slot="button"
    v-bind="$attrs"
  >
    <slot />
  </component>
</template>

<script setup lang="ts">
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from './utils';

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-primary-500/20 focus-visible:ring-[3px]",
  {
    variants: {
      variant: {
        // Primary - metallic gradient
        default: 'btn-metal btn-primary',
        
        // Destructive - keep functional but update colors
        destructive:
          'bg-status-error text-white hover:bg-status-error-dark focus-visible:ring-status-error/20',
          
        // Outline - metallic border
        outline:
          'border border-steel-medium bg-background-secondary text-text-primary hover:bg-background-hover',
          
        // Secondary - metallic but muted
        secondary: 'btn-metal',
        
        // Ghost - subtle hover
        ghost: 'hover:bg-background-hover text-text-primary',
        
        // Link - keep simple
        link: 'text-primary-400 underline-offset-4 hover:underline hover:text-primary-300',
      },
      size: {
        default: 'h-9 px-4 py-2 has-[>svg]:px-3',
        sm: 'h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-10 rounded-md px-6 has-[>svg]:px-4',
        icon: 'size-9 rounded-md',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

interface Props {
  variant?: VariantProps<typeof buttonVariants>['variant'];
  size?: VariantProps<typeof buttonVariants>['size'];
  asChild?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
  asChild: false,
});
</script>