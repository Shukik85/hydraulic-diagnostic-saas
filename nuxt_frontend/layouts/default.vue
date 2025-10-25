<script setup lang="ts">
const isClient = typeof window !== 'undefined'

let authStore: any = null
let colorMode: any = { preference: 'light' } // безопасный дефолт для SSR

if (isClient) {
  try { authStore = useAuthStore() } catch (e) { authStore = { isAuthenticated: false } }
  try { colorMode = useColorMode() } catch (e) { colorMode = { preference: 'light' } }
}

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
  const acc: { name: string; href: string }[] = [{ name: 'Главная', href: '/' }]
  let current = ''
  for (const p of parts) {
    current += `/${p}`
    acc.push({ name: mapName(current), href: current })
  }
  return acc
})

function toggleTheme() {
  if (!isClient) return
  colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'
}

</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Solid Top bar without transparency or flicker -->
    <div class="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-90 transition-opacity duration-200">
            <div class="w-8 h-8 bg-linear-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]" />
            </div>
            <div>
              <h1 class="text-lg font-bold text-gray-900 dark:text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.25)] dark:drop-shadow-[0_1px_1px_rgba(0,0,0,0.6)]">Гидравлика ИИ</h1>
              <p class="text-xs text-gray-700 dark:text-gray-300 leading-tight drop-shadow-[0_1px_1px_rgba(0,0,0,0.15)] dark:drop-shadow-[0_1px_1px_rgba(0,0,0,0.5)]">Промышленный мониторинг</p>
            </div>
          </NuxtLink>

          <div class="flex items-center space-x-3">
            <button @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'" class="p-2 text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors">
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5 drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]" />
            </button>
            <NuxtLink
              to="/dashboard"
              class="px-6 py-2.5 text-sm font-bold text-white bg-linear-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              Открыть дашборд
            </NuxtLink>
          </div>
        </div>

        <!-- Softer Breadcrumbs -->
        <nav class="flex items-center space-x-2 text-sm py-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900">
          <Icon name="heroicons:home" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="font-medium text-gray-700 hover:text-blue-700 dark:text-gray-300 dark:hover:text-blue-200 transition-colors duration-200 hover:underline hover:bg-blue-50 dark:hover:bg-blue-900/30 px-2 py-1 rounded-md"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="font-semibold text-gray-900 dark:text-white bg-blue-50/50 dark:bg-blue-800/30 px-2.5 py-1 rounded-md border border-blue-100/50 dark:border-blue-700/40">
              {{ crumb.name }}
            </span>
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
