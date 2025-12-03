<script setup lang="ts">
import { computed } from 'vue';

const uiStore = useUiStore();

const isDark = computed(() => uiStore.theme === 'dark');

const toggleTheme = () => {
  uiStore.setTheme(isDark.value ? 'light' : 'dark');
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
