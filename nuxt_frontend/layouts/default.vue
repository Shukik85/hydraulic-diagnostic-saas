<script setup lang="ts">
const route = useRoute()
const isDark = ref(false)

// Simple theme toggle without stores
const toggleTheme = () => {
  isDark.value = !isDark.value
  if (process.client) {
    document.documentElement.classList.toggle('dark', isDark.value)
  }
}

// Initialize theme on client
onMounted(() => {
  const stored = localStorage.getItem('color-mode')
  isDark.value = stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', isDark.value)
})
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Clean Static Navbar for Landing -->
    <nav class="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div class="container mx-auto flex items-center justify-between h-16 px-6">
        <!-- Logo with breathing room -->
        <NuxtLink to="/" class="flex items-center space-x-3 mr-12 group">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
          </div>
          <div>
            <span class="text-lg font-bold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
              Гидравлика ИИ
            </span>
            <span class="block text-xs text-gray-500 dark:text-gray-400 leading-tight">
              Диагностическая платформа
            </span>
          </div>
        </NuxtLink>

        <!-- Simple public navigation -->
        <div class="hidden md:flex items-center space-x-8">
          <NuxtLink 
            to="/about" 
            class="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors font-medium"
          >
            О платформе
          </NuxtLink>
          <NuxtLink 
            to="/pricing" 
            class="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors font-medium"
          >
            Тарифы
          </NuxtLink>
          <NuxtLink 
            to="/contact" 
            class="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors font-medium"
          >
            Контакты
          </NuxtLink>
        </div>

        <!-- Clean actions -->
        <div class="flex items-center space-x-4">
          <button 
            @click="toggleTheme" 
            class="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            title="Переключить тему"
          >
            <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
          </button>
          <NuxtLink 
            to="/dashboard" 
            class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-200"
          >
            Открыть дашборд
          </NuxtLink>
        </div>

        <!-- Mobile menu button -->
        <button class="md:hidden p-2 text-gray-600 dark:text-gray-400 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
          <Icon name="heroicons:bars-3" class="w-6 h-6" />
        </button>
      </div>
    </nav>

    <!-- Main Content - NO breadcrumbs on landing -->
    <main>
      <slot />
    </main>
  </div>
</template>