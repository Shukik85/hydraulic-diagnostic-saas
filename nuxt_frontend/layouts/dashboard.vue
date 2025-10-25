<script setup lang="ts">
// Enhanced dashboard layout with better contrast and full Russian localization
const authStore = useAuthStore()
const colorMode = useColorMode()

// Navigation items with full Russian translation
const navigationItems = computed(() => {
  const items = [
    {
      name: 'Обзор',
      href: '/dashboard',
      icon: 'heroicons:squares-2x2',
      description: 'Главный дашборд и метрики'
    },
    {
      name: 'Системы',
      href: '/systems',
      icon: 'heroicons:server-stack',
      description: 'Мониторинг гидравлических систем'
    },
    {
      name: 'Диагностика',
      href: '/diagnostics',
      icon: 'heroicons:cpu-chip',
      description: 'ИИ-анализ и диагностика'
    },
    {
      name: 'Отчёты',
      href: '/reports',
      icon: 'heroicons:document-text',
      description: 'Отчёты и аналитика'
    },
    {
      name: 'ИИ Чат',
      href: '/chat',
      icon: 'heroicons:chat-bubble-left-ellipsis',
      description: 'Интеллектуальный помощник'
    },
    {
      name: 'Настройки',
      href: '/settings',
      icon: 'heroicons:cog-6-tooth',
      description: 'Конфигурация системы'
    }
  ]
  
  // Add investor dashboard for admin/investor role
  if (authStore.user?.role === 'admin' || authStore.user?.role === 'operator') {
    items.push({
      name: 'Бизнес-аналитика',
      href: '/investors',
      icon: 'heroicons:presentation-chart-line',
      description: 'Показатели для руководства'
    })
  }
  
  return items
})

// Mobile menu state
const isMobileMenuOpen = ref(false)

// User menu items
const userMenuItems = [
  { name: 'Профиль', href: '/profile', icon: 'heroicons:user' },
  { name: 'Уведомления', href: '/notifications', icon: 'heroicons:bell' },
  { name: 'Помощь', href: '/help', icon: 'heroicons:question-mark-circle' }
]

const handleLogout = async () => {
  await authStore.logout()
  await navigateTo('/auth/login')
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Mobile menu overlay -->
    <div 
      v-if="isMobileMenuOpen"
      class="fixed inset-0 z-40 lg:hidden"
      @click="isMobileMenuOpen = false"
    >
      <div class="fixed inset-0 bg-gray-600/75 transition-opacity"></div>
    </div>

    <!-- Mobile sidebar -->
    <div 
      :class="[
        'fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-xl transform transition-transform lg:hidden',
        isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
      ]"
    >
      <div class="flex h-full flex-col">
        <!-- Mobile header -->
        <div class="flex h-16 items-center justify-between px-4 border-b border-gray-200 dark:border-gray-700">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">Меню</h2>
          <button
            @click="isMobileMenuOpen = false"
            class="rounded-lg p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <Icon name="heroicons:x-mark" class="h-6 w-6" />
          </button>
        </div>
        
        <!-- Mobile navigation -->
        <nav class="flex-1 space-y-1 px-3 py-4">
          <NuxtLink
            v-for="item in navigationItems"
            :key="item.name"
            :to="item.href"
            @click="isMobileMenuOpen = false"
            :class="[
              'group flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors',
              $route.path === item.href
                ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                : 'text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700'
            ]"
          >
            <Icon :name="item.icon" class="mr-3 h-5 w-5 flex-shrink-0" />
            {{ item.name }}
          </NuxtLink>
        </nav>
      </div>
    </div>

    <!-- Desktop sidebar -->
    <div class="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
      <div class="flex grow flex-col gap-y-5 overflow-y-auto bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
        <!-- Logo -->
        <div class="flex h-16 shrink-0 items-center px-6 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 class="text-lg font-bold text-gray-900 dark:text-white">
                Гидравлика ИИ
              </h1>
              <p class="text-xs text-gray-500 dark:text-gray-400">
                Диагностическая платформа
              </p>
            </div>
          </div>
        </div>
        
        <!-- Navigation with improved contrast -->
        <nav class="flex flex-1 flex-col px-4">
          <ul role="list" class="flex flex-1 flex-col gap-y-2">
            <li v-for="item in navigationItems" :key="item.name">
              <NuxtLink
                :to="item.href"
                :class="[
                  'group flex gap-x-3 rounded-lg p-3 text-sm font-semibold leading-6 transition-all duration-200',
                  $route.path === item.href
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg'
                    : 'text-gray-700 hover:text-blue-600 hover:bg-blue-50 dark:text-gray-200 dark:hover:text-blue-400 dark:hover:bg-blue-900/30'
                ]"
              >
                <Icon 
                  :name="item.icon" 
                  :class="[
                    'h-6 w-6 shrink-0 transition-colors',
                    $route.path === item.href
                      ? 'text-blue-100'
                      : 'text-gray-500 group-hover:text-blue-600 dark:text-gray-400 dark:group-hover:text-blue-400'
                  ]" 
                />
                <div class="flex-1">
                  <div class="font-semibold">{{ item.name }}</div>
                  <div :class="[
                    'text-xs mt-0.5',
                    $route.path === item.href
                      ? 'text-blue-100'
                      : 'text-gray-500 group-hover:text-blue-500 dark:text-gray-400 dark:group-hover:text-blue-300'
                  ]">
                    {{ item.description }}
                  </div>
                </div>
              </NuxtLink>
            </li>
          </ul>
          
          <!-- User section with better contrast -->
          <div class="mt-auto pb-4">
            <div class="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/30 rounded-xl p-4 mb-4">
              <div class="flex items-center space-x-3 mb-3">
                <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <Icon name="heroicons:user" class="w-5 h-5 text-white" />
                </div>
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-semibold text-gray-900 dark:text-white truncate">
                    {{ authStore.userName || 'Пользователь' }}
                  </p>
                  <p class="text-xs text-gray-500 dark:text-gray-400 truncate">
                    {{ authStore.user?.email || 'email@domain.com' }}
                  </p>
                </div>
              </div>
              
              <div class="grid grid-cols-2 gap-2">
                <button
                  @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'"
                  class="flex items-center justify-center space-x-1 px-3 py-2 text-xs font-medium rounded-lg bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
                >
                  <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-3 h-3" />
                  <span>{{ colorMode.preference === 'dark' ? 'Светлая' : 'Тёмная' }}</span>
                </button>
                
                <button
                  @click="handleLogout"
                  class="flex items-center justify-center space-x-1 px-3 py-2 text-xs font-medium rounded-lg bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors"
                >
                  <Icon name="heroicons:arrow-right-on-rectangle" class="w-3 h-3" />
                  <span>Выход</span>
                </button>
              </div>
            </div>
          </div>
        </nav>
      </div>
    </div>

    <!-- Main content -->
    <div class="lg:pl-72">
      <!-- Top bar -->
      <div class="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-4 sm:gap-x-6 sm:px-6 lg:px-8">
        <button
          type="button"
          class="-m-2.5 p-2.5 text-gray-700 dark:text-gray-300 lg:hidden"
          @click="isMobileMenuOpen = true"
        >
          <Icon name="heroicons:bars-3" class="h-6 w-6" />
        </button>
        
        <div class="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
          <div class="flex flex-1 items-center">
            <div class="w-full max-w-md">
              <label class="sr-only">Поиск</label>
              <div class="relative">
                <div class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                  <Icon name="heroicons:magnifying-glass" class="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="search"
                  placeholder="Поиск по системам, отчётам..."
                  class="block w-full rounded-lg border-0 py-2 pl-10 pr-3 text-gray-900 dark:text-white bg-white/60 dark:bg-gray-700/60 ring-1 ring-inset ring-gray-300 dark:ring-gray-600 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-blue-600 sm:text-sm"
                />
              </div>
            </div>
          </div>
          
          <div class="flex items-center gap-x-4 lg:gap-x-6">
            <!-- Notifications -->
            <button class="-m-2.5 p-2.5 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 relative">
              <Icon name="heroicons:bell" class="h-6 w-6" />
              <span class="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 flex items-center justify-center">
                <span class="text-xs font-medium text-white">3</span>
              </span>
            </button>
            
            <!-- Quick actions -->
            <PremiumButton 
              to="/diagnostics" 
              size="sm" 
              icon="heroicons:plus" 
              gradient
              class="hidden sm:inline-flex"
            >
              Новая диагностика
            </PremiumButton>
          </div>
        </div>
      </div>
      
      <!-- Page content -->
      <main class="py-8">
        <slot />
      </main>
    </div>
  </div>
</template>