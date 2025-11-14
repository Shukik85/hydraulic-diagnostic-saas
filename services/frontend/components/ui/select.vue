<template>
  <div class="relative">
    <button
      ref="triggerRef"
      :class="[
        'flex w-full items-center justify-between gap-2 rounded-md border border-steel-medium',
        'bg-steel-dark px-3 py-2 text-sm text-text-primary',
        'transition-colors focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-background-primary',
        'hover:border-primary-400 hover:bg-steel-darker',
        'disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'h-8' : 'h-9',
        className
      ]"
      @click="toggle"
      v-bind="$attrs"
    >
      <span v-if="modelValue" class="flex items-center gap-2">
        <slot name="selected" :value="modelValue">{{ modelValue }}</slot>
      </span>
      <span v-else class="text-text-secondary">
        <slot name="placeholder">{{ placeholder }}</slot>
      </span>
      <Icon name="lucide:chevron-down" class="h-4 w-4 text-steel-light" />
    </button>
    <Transition
      enter-active-class="transition-all duration-200"
      enter-from-class="opacity-0 scale-95"
      enter-to-class="opacity-100 scale-100"
      leave-active-class="transition-all duration-200"
      leave-from-class="opacity-100 scale-100"
      leave-to-class="opacity-0 scale-95"
    >
      <div
        v-if="isOpen"
        ref="contentRef"
        :class="[
          'absolute z-50 bg-steel-darker text-text-primary rounded-md border border-steel-medium shadow-lg p-1 min-w-32 max-h-60 overflow-y-auto',
          contentClass
        ]"
        :style="{ top: position.top + 'px', left: position.left + 'px' }"
      >
        <slot />
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
// ...Остальной код без изменений...
</script>