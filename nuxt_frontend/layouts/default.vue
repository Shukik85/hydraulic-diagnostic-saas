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
    <!-- Top bar -->
    <div class="border-b border-gray-200 dark:border-gray-700 bg-white/96 dark:bg-gray-800/96 backdrop-blur-md shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-90">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
            </div>
            <h1 class="text-lg font-bold text-gray-900 dark:text-white">Гидравлика ИИ</h1>
          </NuxtLink>
          <div class="flex items-center space-x-3">
            <NuxtLink
              to="/dashboard"
              class="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
            >
              Дашборд
            </NuxtLink>
          </div>
        </div>
        <!-- Breadcrumbs -->
        <nav class="flex items-center space-x-2 text-sm py-2">
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="text-gray-900 dark:text-white font-medium">{{ crumb.name }}</span>
            <Icon v-if="i < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-400 dark:text-gray-500" />
          </template>
        </nav>
      </div>
    </div>

    <main>
      <slot />
    </main>
  </div>
</template>
