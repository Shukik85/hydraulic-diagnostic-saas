<script setup lang="ts">
import { computed } from 'vue';
import { useUiStore } from '~/stores/ui';

const uiStore = useUiStore();

const isDark = computed(() => {
  if (uiStore.theme === 'auto') {
    // Check system preference
    if (import.meta.client) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  }
  return uiStore.theme === 'dark';
});

const toggleTheme = () => {
  const newTheme = isDark.value ? 'light' : 'dark';
  console.log('Theme switching:', uiStore.theme, '->', newTheme);
  uiStore.setTheme(newTheme);
};
</script>

<template>
  <button
    type="button"
    :aria-label="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
    class="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-background text-foreground transition-all hover:bg-secondary hover:border-primary/50"
    @click="toggleTheme"
  >
    <Icon
      :name="isDark ? 'heroicons:sun' : 'heroicons:moon'"
      class="h-5 w-5 transition-transform duration-300"
      :class="{ 'rotate-180': isDark }"
    />
  </button>
</template>
