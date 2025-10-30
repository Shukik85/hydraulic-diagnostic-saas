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
  <div class="min-h-screen bg-gray-50">
    <!-- Navigation -->
    <nav class="sticky top-0 z-50 bg-white/95 backdrop-blur-sm border-b border-gray-200 shadow-sm">
      <div class="u-container u-flex-between h-16">
        <NuxtLink to="/" class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
          </div>
          <div>
            <span class="text-lg font-bold text-gray-900">Hydraulic Diagnostic</span>
            <span class="block text-xs text-gray-500 leading-tight">AI Platform</span>
          </div>
        </NuxtLink>

        <div class="hidden md:flex items-center space-x-8">
          <NuxtLink 
            to="/dashboard" 
            class="text-gray-700 hover:text-blue-700 u-transition-fast font-medium"
          >
            Dashboard
          </NuxtLink>
          <NuxtLink 
            to="/diagnostics" 
            class="text-gray-700 hover:text-blue-700 u-transition-fast font-medium"
          >
            Diagnostics
          </NuxtLink>
          <NuxtLink 
            to="/reports" 
            class="text-gray-700 hover:text-blue-700 u-transition-fast font-medium"
          >
            Reports
          </NuxtLink>
          <!-- Help replaces Search -->
          <NuxtLink 
            to="/chat" 
            class="flex items-center gap-2 text-gray-700 hover:text-blue-700 u-transition-fast font-medium"
          >
            <Icon name="heroicons:question-mark-circle" class="w-4 h-4" />
            Help
          </NuxtLink>
        </div>

        <div class="flex items-center gap-4">
          <button
            @click="toggleTheme"
            class="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg u-transition-fast"
            title="Toggle theme"
          >
            <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
          </button>
          
          <!-- Profile with Settings in dropdown -->
          <UDropdown :items="[[{ label: 'Profile', to: '/profile' }, { label: 'Settings', to: '/settings' }], [{ label: 'Logout', to: '/auth/login' }]]">
            <button class="flex items-center gap-2 p-1.5 rounded-lg hover:bg-gray-100">
              <div class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                <Icon name="heroicons:user" class="w-4 h-4 text-white" />
              </div>
              <Icon name="heroicons:chevron-down" class="w-4 h-4 text-gray-600" />
            </button>
          </UDropdown>
        </div>
      </div>
    </nav>

    <main class="u-container py-8">
      <slot />
    </main>
  </div>
</template>

<style scoped>
</style>