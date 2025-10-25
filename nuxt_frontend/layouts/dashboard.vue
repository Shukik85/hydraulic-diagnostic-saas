<script setup lang="ts">
const authStore = useAuthStore()
const colorMode = useColorMode()
const route = useRoute()

// Navigation items
const navigationItems = computed(() => {
  const items = [
    { name: 'Главная', href: '/', icon: 'heroicons:home', description: 'Перейти на главную страницу', isExternal: true },
    { name: 'Обзор', href: '/dashboard', icon: 'heroicons:squares-2x2', description: 'Главный дашборд и метрики' },
    { name: 'Системы', href: '/systems', icon: 'heroicons:server-stack', description: 'Мониторинг гидравлических систем' },
    { name: 'Диагностика', href: '/diagnostics', icon: 'heroicons:cpu-chip', description: 'ИИ-анализ и диагностика' },
    { name: 'Отчёты', href: '/reports', icon: 'heroicons:document-text', description: 'Отчёты и аналитика' },
    { name: 'ИИ Чат', href: '/chat', icon: 'heroicons:chat-bubble-left-ellipsis', description: 'Интеллектуальный помощник' },
    { name: 'Настройки', href: '/settings', icon: 'heroicons:cog-6-tooth', description: 'Конфигурация системы' }
  ]
  
  if (authStore.user?.role === 'admin' || authStore.user?.role === 'operator') {
    items.push({
      name: 'Бизнес-аналитика', href: '/investors', icon: 'heroicons:presentation-chart-line', description: 'Показатели для руководства'
    })
  }
  
  return items
})

// Breadcrumbs with improved mapping
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
  const acc: { name: string, href: string }[] = [{ name: 'Главная', href: '/' }]
  let current = ''
  for (const p of parts) {
    current += `/${p}`
    acc.push({ name: mapName(current), href: current })
  }
  return acc
})

const isMobileMenuOpen = ref(false)

const handleLogout = async () => {
  await authStore.logout()
  await navigateTo('/auth/login')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Mobile menu overlay -->
    <div v-if="isMobileMenuOpen" class="fixed inset-0 z-40 lg:hidden" @click="isMobileMenuOpen = false">
      <div class="fixed inset-0 bg-gray-600/75 transition-opacity"></div>
    </div>

    <!-- Mobile sidebar -->
    <div :class="[
      'fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-xl transform transition-transform lg:hidden',
      isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
    ]">
      <div class="flex h-full flex-col">
        <div class="flex h-16 items-center justify-between px-4 border-b border-gray-200 dark:border-gray-700">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100">Навигация</h2>
          <button @click="isMobileMenuOpen = false" class="rounded-lg p-1 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
            <Icon name="heroicons:x-mark" class="h-6 w-6" />
          </button>
        </div>
        <nav class="flex-1 space-y-1 px-3 py-4">
          <template v-for="item in navigationItems" :key="item.name">
            <NuxtLink v-if="!item.isExternal" :to="item.href" @click="isMobileMenuOpen = false" :class="[
              'group flex items-center rounded-lg px-3 py-3 text-sm font-medium transition-colors hover:underline',
              $route.path === item.href ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md' : 'text-gray-900 hover:bg-blue-100 hover:text-blue-700 dark:text-gray-100 dark:hover:bg-blue-800/40 dark:hover:text-blue-200'
            ]">
              <Icon :name="item.icon" class="mr-3 h-5 w-5 flex-shrink-0" />
              {{ item.name }}
            </NuxtLink>
            <NuxtLink v-else :to="item.href" class="w-full group flex items-center rounded-lg px-3 py-3 text-sm font-medium transition-colors text-gray-900 hover:bg-green-100 hover:text-green-700 dark:text-gray-100 dark:hover:bg-green-800/40 dark:hover:text-green-200 border border-dashed border-green-200 dark:border-green-700 hover:underline">
              <Icon :name="item.icon" class="mr-3 h-5 w-5 flex-shrink-0 text-green-600 dark:text-green-400" />
              {{ item.name }} <Icon name="heroicons:arrow-top-right-on-square" class="w-3 h-3 ml-auto" />
            </NuxtLink>
          </template>
        </nav>
      </div>
    </div>

    <!-- Desktop sidebar -->
    <div class="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-80 lg:flex-col">
      <div class="flex grow flex-col gap-y-5 overflow-y-auto bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 shadow-lg">
        <div class="flex h-16 shrink-0 items-center px-6 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-blue-900/30">
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-80 transition-opacity">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <Icon name="heroicons:cpu-chip" class="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 class="text-xl font-bold text-gray-900 dark:text-white">Гидравлика ИИ</h1>
              <p class="text-xs text-gray-600 dark:text-gray-400">Диагностическая платформа</p>
            </div>
          </NuxtLink>
        </div>
        
        <nav class="flex flex-1 flex-col px-4">
          <ul role="list" class="flex flex-1 flex-col gap-y-2">
            <li v-for="item in navigationItems" :key="item.name">
              <NuxtLink v-if="!item.isExternal" :to="item.href" :class="[
                'group flex gap-x-3 rounded-xl p-4 text-sm font-semibold leading-6 transition-all duration-200 hover:underline',
                $route.path === item.href
                  ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg'
                  : 'text-gray-900 hover:text-blue-700 hover:bg-blue-100 dark:text-gray-100 dark:hover:text-blue-200 dark:hover:bg-blue-800/40'
              ]">
                <Icon :name="item.icon" class="h-6 w-6 shrink-0" />
                <div class="flex-1">
                  <div class="font-bold">{{ item.name }}</div>
                  <div class="text-xs mt-1 opacity-75">{{ item.description }}</div>
                </div>
              </NuxtLink>
              <NuxtLink v-else :to="item.href" class="group flex gap-x-3 rounded-xl p-4 text-sm font-semibold leading-6 transition-all duration-200 hover:underline text-gray-900 hover:text-green-700 hover:bg-green-100 dark:text-gray-100 dark:hover:text-green-200 dark:hover:bg-green-800/40 border-2 border-dashed border-green-200 dark:border-green-700">
                <Icon :name="item.icon" class="h-6 w-6 shrink-0 text-green-600 dark:text-green-400" />
                <div class="flex-1">
                  <div class="font-bold flex items-center">
                    {{ item.name }}
                    <Icon name="heroicons:arrow-top-right-on-square" class="w-3 h-3 ml-2" />
                  </div>
                  <div class="text-xs mt-1 opacity-75">{{ item.description }}</div>
                </div>
              </NuxtLink>
            </li>
          </ul>
          
          <div class="mt-auto pb-4">
            <div class="bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50 dark:from-gray-800 dark:via-blue-900/30 dark:to-indigo-900/30 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
              <div class="flex items-center space-x-3 mb-3">
                <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-md">
                  <Icon name="heroicons:user" class="w-6 h-6 text-white" />
                </div>
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-bold text-gray-900 dark:text-white truncate">{{ authStore.userName || 'Пользователь' }}</p>
                  <p class="text-xs text-gray-700 dark:text-gray-300 truncate">{{ authStore.user?.email || 'email@domain.com' }}</p>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-2">
                <button @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'" class="flex items-center justify-center space-x-2 px-3 py-2 text-xs font-semibold rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors shadow-sm border border-gray-200 dark:border-gray-600">
                  <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-4 h-4" />
                  {{ colorMode.preference === 'dark' ? 'Светлая' : 'Тёмная' }}
                </button>
                <button @click="handleLogout" class="flex items-center justify-center space-x-2 px-3 py-2 text-xs font-semibold rounded-lg bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors shadow-sm border border-red-200 dark:border-red-700">
                  <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4" />
                  Выход
                </button>
              </div>
            </div>
          </div>
        </nav>
      </div>
    </div>

    <!-- Main content -->
    <div class="lg:pl-80">
      <!-- Top bar with breadcrumbs -->
      <div class="sticky top-0 z-40 bg-white/96 dark:bg-gray-900/96 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div class="flex h-16 items-center gap-x-4 px-4 sm:gap-x-6 sm:px-6 lg:px-8">
          <button type="button" class="-m-2.5 p-2.5 text-gray-900 dark:text-gray-100 lg:hidden hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors" @click="isMobileMenuOpen = true">
            <Icon name="heroicons:bars-3" class="h-6 w-6" />
          </button>
          
          <!-- Maximum Contrast Breadcrumbs -->
          <nav class="flex items-center space-x-2 text-sm flex-1" aria-label="Breadcrumb">
            <template v-for="(crumb, index) in breadcrumbs" :key="crumb.href">
              <NuxtLink v-if="index < breadcrumbs.length - 1" :to="crumb.href" class="font-medium text-gray-700 hover:text-blue-700 dark:text-gray-300 dark:hover:text-blue-200 transition-colors duration-200 hover:underline hover:bg-blue-50 dark:hover:bg-blue-900/30 px-2 py-1 rounded-md">
                {{ crumb.name }}
              </NuxtLink>
              <span v-else class="font-bold text-gray-900 dark:text-white bg-blue-100 dark:bg-blue-800/50 px-3 py-1 rounded-lg border border-blue-200 dark:border-blue-700">{{ crumb.name }}</span>
              <Icon v-if="index < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
            </template>
          </nav>
          
          <!-- Actions -->
          <div class="flex items-center gap-x-4">
            <button class="relative p-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
              <Icon name="heroicons:bell" class="h-5 w-5" />
              <span class="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 flex items-center justify-center animate-pulse">
                <span class="text-xs font-bold text-white">3</span>
              </span>
            </button>
            <button class="hidden sm:inline-flex items-center gap-x-2 px-4 py-2 text-sm font-semibold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 hover:bg-blue-100 transition-all duration-200 shadow-lg hover:shadow-xl" @click="navigateTo('/diagnostics')">
              <Icon name="heroicons:plus" class="w-4 h-4" />
              Новая диагностика
            </button>
          </div>
        </div>
      </div>
      
      <main>
        <slot />
      </main>
    </div>
  </div>
</template>