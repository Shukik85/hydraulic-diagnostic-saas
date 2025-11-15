<template>
  <button
    :class="[
      'inline-flex items-center justify-center gap-2 rounded-lg text-sm font-bold select-none transition-all duration-200 outline-none',
      variant === 'outline'
        ? 'border border-steel-medium bg-transparent hover:bg-steel-dark text-primary-400 hover:text-primary-300'
        : 'bg-steel-dark text-text-primary hover:bg-primary-500/10',
      pressed
        ? 'bg-primary-600/90 text-white shadow focus:ring-2 focus:ring-primary-600 focus:ring-offset-2'
        : '',
      size === 'sm' ? 'h-8 px-2 min-w-8' : size === 'lg' ? 'h-10 px-4 min-w-10' : 'h-9 px-3 min-w-9',
      'disabled:opacity-50 disabled:cursor-not-allowed',
      className
    ]"
    :data-state="pressed ? 'on' : 'off'"
    @click="toggle"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>
<script setup lang="ts">// ...Остальной код без изменений...</script>