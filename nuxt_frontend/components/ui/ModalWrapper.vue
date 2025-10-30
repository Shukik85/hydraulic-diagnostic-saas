<template>
  <ClientOnly>
    <Teleport to="body" :disabled="!mounted">
      <slot />
    </Teleport>
    <template #fallback>
      <!-- Fallback for SSR -->
      <div></div>
    </template>
  </ClientOnly>
</template>

<script setup lang="ts">
/**
 * Wrapper component to ensure proper Teleport functionality
 * across all pages and prevent SSR/SPA conflicts
 */

const mounted = ref(false)

onMounted(() => {
  // Ensure DOM is ready before teleporting
  nextTick(() => {
    mounted.value = true
  })
})

onBeforeUnmount(() => {
  mounted.value = false
})
</script>