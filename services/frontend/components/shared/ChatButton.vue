<script setup lang="ts">
import { computed } from 'vue';

const emit = defineEmits<{
  openChat: [];
}>();

// TODO: Replace with actual unread count from RAG store
const unreadCount = computed(() => 0);

const handleClick = () => {
  emit('openChat');
};

// Keyboard shortcut: Cmd+K or Ctrl+K
onMounted(() => {
  const handleKeydown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      handleClick();
    }
  };
  window.addEventListener('keydown', handleKeydown);
  onUnmounted(() => window.removeEventListener('keydown', handleKeydown));
});
</script>

<template>
  <button
    type="button"
    aria-label="Open AI Assistant (Cmd+K)"
    class="relative flex h-9 items-center gap-2 rounded-lg border border-border bg-background px-3 text-sm text-muted-foreground transition-all hover:bg-secondary hover:border-primary/50 hover:text-foreground"
    @click="handleClick"
  >
    <Icon name="heroicons:magnifying-glass" class="h-4 w-4" />
    <span class="hidden sm:inline">Search or ask AI...</span>
    <kbd class="hidden sm:inline-flex h-5 items-center gap-1 rounded border border-border bg-muted px-1.5 text-xs font-medium text-muted-foreground">
      <span class="text-xs">⌘</span>K
    </kbd>
    
    <!-- Unread badge -->
    <span
      v-if="unreadCount > 0"
      class="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground"
    >
      {{ unreadCount > 9 ? '9+' : unreadCount }}
    </span>
  </button>
</template>
