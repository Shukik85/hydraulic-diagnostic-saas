<script setup lang="ts">
import { computed } from 'vue';

interface CardProps {
  variant?: 'default' | 'outlined' | 'elevated' | 'glass';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hoverable?: boolean;
  clickable?: boolean;
  bordered?: boolean;
  rounded?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  shadow?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  as?: 'div' | 'article' | 'section';
}

const props = withDefaults(defineProps<CardProps>(), {
  variant: 'default',
  padding: 'md',
  hoverable: false,
  clickable: false,
  bordered: true,
  rounded: 'lg',
  shadow: 'md',
  as: 'div',
});

const emit = defineEmits<{
  click: [event: MouseEvent];
}>();

const handleClick = (event: MouseEvent) => {
  if (props.clickable) {
    emit('click', event);
  }
};

const cardClasses = computed(() => {
  const classes = [
    // Base styles
    'bg-white dark:bg-gray-900',
    'transition-all duration-200',
  ];

  // Variant styles
  const variantClasses = {
    default: 'bg-white dark:bg-gray-900',
    outlined: 'bg-transparent border-2',
    elevated: 'bg-white dark:bg-gray-800',
    glass: 'bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm',
  };
  classes.push(variantClasses[props.variant]);

  // Padding
  const paddingClasses = {
    none: 'p-0',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };
  classes.push(paddingClasses[props.padding]);

  // Rounded corners
  const roundedClasses = {
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    xl: 'rounded-xl',
    '2xl': 'rounded-2xl',
  };
  classes.push(roundedClasses[props.rounded]);

  // Shadow
  if (props.shadow !== 'none') {
    const shadowClasses = {
      sm: 'shadow-sm',
      md: 'shadow-md',
      lg: 'shadow-lg',
      xl: 'shadow-xl',
    };
    classes.push(shadowClasses[props.shadow]);
  }

  // Border
  if (props.bordered) {
    classes.push('border border-gray-200 dark:border-gray-700');
  }

  // Hoverable
  if (props.hoverable) {
    classes.push('hover:shadow-lg hover:scale-[1.02]');
  }

  // Clickable
  if (props.clickable) {
    classes.push('cursor-pointer');
  }

  return classes.join(' ');
});

const headerClasses = computed(() => {
  const classes = ['card-header'];
  
  if (props.padding !== 'none') {
    classes.push('border-b border-gray-200 dark:border-gray-700 pb-4 mb-4');
  }
  
  return classes.join(' ');
});

const footerClasses = computed(() => {
  const classes = ['card-footer'];
  
  if (props.padding !== 'none') {
    classes.push('border-t border-gray-200 dark:border-gray-700 pt-4 mt-4');
  }
  
  return classes.join(' ');
});
</script>

<template>
  <component
    :is="as"
    :class="cardClasses"
    :role="clickable ? 'button' : undefined"
    :tabindex="clickable ? 0 : undefined"
    :aria-label="clickable ? 'Card button' : undefined"
    @click="handleClick"
    @keydown.enter="clickable && handleClick"
    @keydown.space.prevent="clickable && handleClick"
  >
    <!-- Header slot -->
    <div v-if="$slots.header" :class="headerClasses">
      <slot name="header" />
    </div>

    <!-- Default content slot -->
    <div class="card-content">
      <slot />
    </div>

    <!-- Footer slot -->
    <div v-if="$slots.footer" :class="footerClasses">
      <slot name="footer" />
    </div>
  </component>
</template>

<style scoped>
/* Additional custom styles if needed */
.card-content {
  /* Ensures proper text flow */
  word-wrap: break-word;
  overflow-wrap: break-word;
}
</style>
