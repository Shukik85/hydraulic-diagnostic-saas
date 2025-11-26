<script setup lang="ts">
import { onMounted } from 'vue';

// Restore user session and theme on app mount
const authStore = useAuthStore();
const uiStore = useUiStore();

onMounted(() => {
  // Restore auth session
  authStore.restoreSession();

  // Restore UI preferences
  uiStore.restoreTheme();
  uiStore.restoreSidebar();
});
</script>

<template>
  <div>
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
  </div>
</template>
