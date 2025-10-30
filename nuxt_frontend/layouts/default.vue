<script setup lang="ts">
const route = useRoute()
const isDark = ref(false)

const toggleTheme = () => {
  isDark.value = !isDark.value
  if (process.client) {
    document.documentElement.classList.toggle('dark', isDark.value)
    localStorage.setItem('color-mode', isDark.value ? 'dark' : 'light')
  }
}

onMounted(() => {
  const stored = localStorage.getItem('color-mode')
  isDark.value =
    stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', isDark.value)
})
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Navigation -->
    <nav class="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div class="u-container u-flex-between h-16">
        <NuxtLink to="/" class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
          </div>
          <div>
            <span class="text-lg font-bold text-gray-900 dark:text-white">Hydraulic Diagnostic</span>
            <span class="block text-xs text-gray-500 dark:text-gray-400 leading-tight">AI Platform</span>
          </div>
        </NuxtLink>

        <div class="hidden md:flex items-center space-x-8">
          <NuxtLink 
            to="/dashboard" 
            class="text-gray-700 dark:text-gray-300 hover:text-blue-700 dark:hover:text-blue-300 u-transition-fast font-medium"
          >
            Dashboard
          </NuxtLink>
          <NuxtLink 
            to="/diagnostics" 
            class="text-gray-700 dark:text-gray-300 hover:text-blue-700 dark:hover:text-blue-300 u-transition-fast font-medium"
          >
            Diagnostics
          </NuxtLink>
          <NuxtLink 
            to="/reports" 
            class="text-gray-700 dark:text-gray-300 hover:text-blue-700 dark:hover:text-blue-300 u-transition-fast font-medium"
          >
            Reports
          </NuxtLink>
          <NuxtLink 
            to="/settings" 
            class="text-gray-700 dark:text-gray-300 hover:text-blue-700 dark:hover:text-blue-300 u-transition-fast font-medium"
          >
            Settings
          </NuxtLink>
        </div>

        <div class="flex items-center gap-4">
          <button
            @click="toggleTheme"
            class="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg u-transition-fast"
            title="Toggle theme"
          >
            <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
          </button>
          
          <div class="flex items-center gap-2">
            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <Icon name="heroicons:user" class="w-4 h-4 text-white" />
            </div>
            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Admin</span>
          </div>
        </div>
      </div>
    </nav>

    <main class="u-container py-8">
      <slot />
    </main>
  </div>
</template>

<style scoped>
/* Remove all @apply to fix Tailwind v4 compatibility */
</style>