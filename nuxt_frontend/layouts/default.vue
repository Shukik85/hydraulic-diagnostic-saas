<script setup lang="ts">
const route = useRoute()

// Breadcrumbs mapping
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

// Navigation items for public layout
const publicNavItems = [
  { to: '/', label: 'Главная', icon: 'heroicons:home' },
  { to: '/about', label: 'О платформе', icon: 'heroicons:information-circle' },
  { to: '/features', label: 'Возможности', icon: 'heroicons:star' },
  { to: '/pricing', label: 'Тарифы', icon: 'heroicons:currency-dollar' },
  { to: '/contact', label: 'Контакты', icon: 'heroicons:phone' }
]

// Event handlers
const handleSearch = () => {
  // TODO: Открыть command palette
  console.log('Opening search...')
}

const handleNotifications = () => {
  // TODO: Открыть панель уведомлений
  console.log('Opening notifications...')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Enhanced Navbar Component -->
    <AppNavbar 
      :items="publicNavItems"
      :show-search="true"
      :show-notifications="false"
      :show-profile="false"
      @open-search="handleSearch"
      @open-notifications="handleNotifications"
    >
      <template #cta>
        <NuxtLink 
          to="/auth/login" 
          class="hidden sm:inline-flex px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          Войти
        </NuxtLink>
        <NuxtLink 
          to="/dashboard" 
          class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-200"
        >
          Открыть дашборд
        </NuxtLink>
      </template>
    </AppNavbar>

    <!-- Breadcrumbs Section -->
    <div class="pt-16">
      <div v-if="route.path !== '/'" class="bg-gray-50 dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800">
        <div class="container mx-auto px-4">
          <nav class="flex items-center space-x-2 text-sm py-3">
            <Icon name="heroicons:home" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
            <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
              <NuxtLink
                v-if="i < breadcrumbs.length - 1"
                :to="crumb.href"
                class="font-medium text-gray-700 hover:text-blue-700 dark:text-gray-300 dark:hover:text-blue-300 transition-colors duration-200 hover:underline hover:bg-blue-50 dark:hover:bg-blue-900/30 px-2 py-1 rounded-md"
              >
                {{ crumb.name }}
              </NuxtLink>
              <span 
                v-else 
                class="font-semibold text-gray-900 dark:text-white bg-blue-50/60 dark:bg-blue-800/30 px-2.5 py-1 rounded-md border border-blue-100/60 dark:border-blue-700/40"
              >
                {{ crumb.name }}
              </span>
              <Icon v-if="i < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </template>
          </nav>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <main :class="route.path === '/' ? 'pt-16' : 'pt-0'">
      <slot />
    </main>
  </div>
</template>