<template>
  <div class="min-h-screen flex items-center justify-center" 
    :class="colorMode?.preference === 'dark' ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 to-indigo-100'">
    <slot />
  </div>
</template>

<script setup lang="ts">
// Safe color mode usage
let colorMode: any = { preference: 'light' }

try {
  colorMode = useColorMode()
} catch (error) {
  // Fallback if color-mode not available
  console.info('Color mode not available, using light theme')
}
</script>