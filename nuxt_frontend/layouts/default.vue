<script setup lang="ts">
const authStore = useAuthStore()
const colorMode = useColorMode()
const route = useRoute()

const mapName = (path: string) => ({
  '/': 'Главная',
  '/dashboard': 'Дашборд',
  '/systems': 'Системы',
  '/diagnostics': 'Диагностика',
  '/reports': 'Отчёты',
  '/chat': 'ИИ Чат',
  '/settings': 'Настройки'
}[path] || 'Страница')

const breadcrumbs = computed(() => {
  const parts = route.path.split('/').filter(Boolean)
  const acc: { name: string, href: string }[] = [{ name: 'Главная', href: '/' }]
  let current = ''
  for (const p of parts) {
    current += `/${p}`
    acc.push({ name: mapName(current), href: current })
  }
  return acc
})
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Maximum Contrast Top bar -->
    <div class="border-b border-gray-200 dark:border-gray-700 bg-white/96 dark:bg-gray-800/96 backdrop-blur-md shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-90 transition-opacity duration-200">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 class="text-lg font-bold text-gray-900 dark:text-white">Гидравлика ИИ</h1>
              <p class="text-xs text-gray-700 dark:text-gray-300 leading-tight">Промышленный мониторинг</p>
            </div>
          </NuxtLink>
          
          <div class="flex items-center space-x-3">
            <button @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'" class="p-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
            </button>
            <NuxtLink
              to="/dashboard"
              class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl hover:bg-blue-100"
            >
              Открыть дашборд
            </NuxtLink>
          </div>
        </div>
        
        <!-- Maximum Contrast Breadcrumbs -->
        <nav class="flex items-center space-x-2 text-sm py-3 border-t border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50">
          <Icon name="heroicons:home" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="font-medium text-gray-700 hover:text-blue-700 dark:text-gray-300 dark:hover:text-blue-200 transition-colors duration-200 hover:underline hover:bg-blue-50 dark:hover:bg-blue-900/30 px-2 py-1 rounded-md"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="font-bold text-gray-900 dark:text-white bg-blue-100 dark:bg-blue-800/50 px-3 py-1 rounded-lg border border-blue-200 dark:border-blue-700">{{ crumb.name }}</span>
            <Icon v-if="i < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
          </template>
        </nav>
      </div>
    </div>

    <main class="py-8">
      <slot />
    </main>
  </div>
</template>