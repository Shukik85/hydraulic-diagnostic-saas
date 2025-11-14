<template>
  <button
    :class="[
      'inline-flex items-center justify-center gap-2 text-sm font-bold select-none outline-none transition-all duration-200',
      'min-w-0 flex-1 shrink-0 rounded-none first:rounded-l-md last:rounded-r-md border border-steel-medium bg-steel shadow focus:z-10 focus-visible:z-10',
      isActive ? 'bg-primary-600 text-white shadow-md shadow-primary-500/40 hover:bg-primary-700' : 'bg-steel-dark text-text-primary hover:bg-primary-600/10',
      variant === 'outline' ? 'hover:bg-steel-medium/20' : '',
      size === 'sm' ? 'h-8 px-2 min-w-8' : size === 'lg' ? 'h-10 px-4 min-w-10' : 'h-9 px-3 min-w-9',
      'disabled:opacity-50 disabled:cursor-not-allowed',
      className
    ]"
    :data-state="isActive ? 'on' : 'off'"
    @click="handleClick"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>
<script setup lang="ts">// ...Остальной код без изменений...</script>