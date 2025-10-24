<script setup lang="ts">
// Dashboard layout with navigation
const authStore = useAuthStore()
const route = useRoute()

// Navigation items based on backend capabilities
const navigation = [
  { name: 'Главная', href: '/', icon: 'heroicons:home', current: route.path === '/' },
  { name: 'Оборудование', href: '/equipment', icon: 'heroicons:server-stack', current: route.path.startsWith('/equipment') },
  { name: 'Диагностика', href: '/diagnostics', icon: 'heroicons:cpu-chip', current: route.path.startsWith('/diagnostics') },
  { name: 'Отчёты', href: '/reports', icon: 'heroicons:document-text', current: route.path.startsWith('/reports') },
  { name: 'Данные сенсоров', href: '/sensors', icon: 'heroicons:signal', current: route.path.startsWith('/sensors') },
  { name: 'AI помощник', href: '/chat', icon: 'heroicons:chat-bubble-left-right', current: route.path.startsWith('/chat') },
  { name: 'Настройки', href: '/settings', icon: 'heroicons:cog-6-tooth', current: route.path.startsWith('/settings') }
]

const sidebarOpen = ref(false)

const handleLogout = async () => {
  await authStore.logout()
}

// Dark mode toggle
const colorMode = useColorMode()
const isDark = computed(() => colorMode.value === 'dark')
const toggleDark = () => {
  colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'
}
</script>

<template>
  <div class="flex h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Sidebar -->
    <div 
      :class="sidebarOpen ? 'translate-x-0' : '-translate-x-full'"
      class="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0"
    >
      <!-- Logo -->
      <div class="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="flex items-center">
          <Icon name="heroicons:wrench-screwdriver" class="w-8 h-8 text-blue-600" />
          <span class="ml-3 text-xl font-bold text-gray-900 dark:text-white">
            Hydraulic SaaS
          </span>
        </div>
        <button 
          @click="sidebarOpen = false"
          class="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <Icon name="heroicons:x-mark" class="w-6 h-6" />
        </button>
      </div>

      <!-- Navigation -->
      <nav class="mt-6 px-3">
        <div class="space-y-1">
          <NuxtLink
            v-for="item in navigation"
            :key="item.name"
            :to="item.href"
            :class="[
              item.current
                ? 'bg-blue-50 border-r-4 border-blue-600 text-blue-700 dark:bg-blue-900/50 dark:border-blue-400 dark:text-blue-300'
                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white',
              'group flex items-center px-3 py-2 text-sm font-medium rounded-l-md transition-colors'
            ]"
          >
            <Icon 
              :name="item.icon" 
              :class="[
                item.current 
                  ? 'text-blue-500 dark:text-blue-400' 
                  : 'text-gray-400 group-hover:text-gray-500 dark:group-hover:text-gray-300',
                'mr-3 flex-shrink-0 w-5 h-5'
              ]"
            />
            {{ item.name }}
          </NuxtLink>
        </div>
      </nav>

      <!-- User menu -->
      <div class="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <span class="text-white font-medium text-sm">
                {{ authStore.userName.charAt(0).toUpperCase() }}
              </span>
            </div>
          </div>
          <div class="ml-3 flex-1 min-w-0">
            <p class="text-sm font-medium text-gray-900 dark:text-white truncate">
              {{ authStore.userName }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400 truncate">
              {{ authStore.user?.email }}
            </p>
          </div>
          <div class="flex items-center space-x-2">
            <button 
              @click="toggleDark"
              class="p-2 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              :title="isDark ? 'Светлая тема' : 'Тёмная тема'"
            >
              <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-4 h-4" />
            </button>
            <button 
              @click="handleLogout"
              class="p-2 text-gray-400 hover:text-red-500"
              title="Выйти"
            >
              <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Mobile sidebar backdrop -->
    <div 
      v-if="sidebarOpen"
      @click="sidebarOpen = false"
      class="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 transition-opacity lg:hidden"
    />

    <!-- Main content -->
    <div class="flex-1 flex flex-col overflow-hidden lg:ml-0">
      <!-- Top header -->
      <header class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between px-4 py-4 sm:px-6">
          <div class="flex items-center">
            <button 
              @click="sidebarOpen = true"
              class="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <Icon name="heroicons:bars-3" class="w-6 h-6" />
            </button>
            
            <!-- Breadcrumb placeholder -->
            <div class="hidden sm:flex sm:items-center sm:space-x-2 sm:ml-4 lg:ml-0">
              <h1 class="text-lg font-medium text-gray-900 dark:text-white">
                {{ 
                  route.meta.title || 
                  navigation.find(n => n.current)?.name || 
                  'Панель управления' 
                }}
              </h1>
            </div>
          </div>

          <div class="flex items-center space-x-4">
            <!-- Notifications placeholder -->
            <button class="p-2 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300">
              <Icon name="heroicons:bell" class="w-6 h-6" />
            </button>
            
            <!-- Health indicator -->
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full"></div>
              <span class="text-sm text-gray-500 dark:text-gray-400 hidden sm:inline">Онлайн</span>
            </div>
          </div>
        </div>
      </header>

      <!-- Main content area -->
      <main class="flex-1 overflow-y-auto">
        <div class="p-6">
          <slot />
        </div>
      </main>
    </div>
  </div>
</template>