<script setup lang="ts">
// Enhanced dashboard with AppNavbar integration
const route = useRoute()

// Dashboard navigation items
const dashboardNavItems = [
  { to: '/', label: 'Главная', icon: 'heroicons:home', external: true },
  { to: '/dashboard', label: 'Обзор', icon: 'heroicons:squares-2x2' },
  { to: '/systems', label: 'Системы', icon: 'heroicons:server-stack' },
  { to: '/diagnostics', label: 'Диагностика', icon: 'heroicons:cpu-chip' },
  { to: '/reports', label: 'Отчёты', icon: 'heroicons:document-text' },
  { to: '/chat', label: 'ИИ Чат', icon: 'heroicons:chat-bubble-left-ellipsis' },
  { to: '/settings', label: 'Настройки', icon: 'heroicons:cog-6-tooth' },
  { to: '/investors', label: 'Бизнес-аналитика', icon: 'heroicons:presentation-chart-line' }
]

// Breadcrumbs
const mapName = (path: string) => ({
  '/dashboard': 'Дашборд',
  '/systems': 'Системы',
  '/diagnostics': 'Диагностика',
  '/reports': 'Отчёты',
  '/chat': 'ИИ Чат',
  '/settings': 'Настройки',
  '/investors': 'Бизнес-аналитика'
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

// Event handlers
const handleSearch = () => {
  console.log('Opening dashboard search...')
}

const handleNotifications = () => {
  console.log('Opening notifications panel...')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Enhanced Navbar for Dashboard -->
    <AppNavbar 
      :items="dashboardNavItems"
      :show-search="true"
      :show-notifications="true"
      :show-profile="true"
      :notifications-count="5"
      @open-search="handleSearch"
      @open-notifications="handleNotifications"
    >
      <template #cta>
        <NuxtLink 
          to="/diagnostics" 
          class="px-5 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg hover:from-green-700 hover:to-emerald-700 shadow-lg hover:shadow-xl transition-all duration-200"
        >
          <Icon name="heroicons:plus" class="w-4 h-4 mr-2 inline" />
          Новая диагностика
        </NuxtLink>
      </template>
    </AppNavbar>

    <!-- Dashboard Breadcrumbs -->
    <div class="pt-16">
      <div class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div class="container mx-auto px-4">
          <nav class="flex items-center space-x-2 text-sm py-4">
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
    <main class="py-6">
      <slot />
    </main>
  </div>
</template>