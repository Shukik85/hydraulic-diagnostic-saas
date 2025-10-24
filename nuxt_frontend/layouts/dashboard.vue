<script setup lang="ts">
// Professional dashboard layout with navigation and user management
const authStore = useAuthStore()
const router = useRouter()
const route = useRoute()

// Navigation items
const navigation = [
  {
    name: 'Обзор',
    href: '/dashboard',
    icon: 'heroicons:home',
    current: route.path === '/dashboard'
  },
  {
    name: 'Оборудование',
    href: '/equipment',
    icon: 'heroicons:wrench-screwdriver',
    current: route.path.startsWith('/equipment')
  },
  {
    name: 'Диагностика',
    href: '/diagnostics',
    icon: 'heroicons:chart-pie',
    current: route.path === '/diagnostics'
  },
  {
    name: 'Отчёты',
    href: '/reports',
    icon: 'heroicons:document-chart-bar',
    current: route.path.startsWith('/reports')
  },
  {
    name: 'Сенсоры',
    href: '/sensors',
    icon: 'heroicons:signal',
    current: route.path === '/sensors'
  },
  {
    name: 'AI-ассистент',
    href: '/chat',
    icon: 'heroicons:chat-bubble-left-right',
    current: route.path === '/chat'
  }
]

// Special navigation for privileged users
const privilegedNavigation = computed(() => {
  const items = []
  
  // Investor dashboard for admins/investors
  if (authStore.user?.role === 'admin' || authStore.user?.role === 'investor') {
    items.push({
      name: 'Бизнес-аналитика',
      href: '/investors',
      icon: 'heroicons:presentation-chart-line',
      current: route.path === '/investors',
      special: true
    })
  }
  
  return items
})

// Mobile menu state
const isMobileMenuOpen = ref(false)
const sidebarOpen = ref(false)

// User menu state
const userMenuOpen = ref(false)

// User menu items
const userNavigation = [
  { name: 'Профиль', href: '/profile', icon: 'heroicons:user' },
  { name: 'Настройки', href: '/settings', icon: 'heroicons:cog-6-tooth' },
  { name: 'Поддержка', href: '/support', icon: 'heroicons:life-buoy' }
]

const handleLogout = async () => {
  try {
    await authStore.logout()
    await navigateTo('/auth/login')
  } catch (error) {
    console.error('Logout error:', error)
  }
}

// Close mobile menu on route change
watch(() => route.path, () => {
  isMobileMenuOpen.value = false
  userMenuOpen.value = false
})

// Close dropdowns when clicking outside
const handleClickOutside = () => {
  userMenuOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <!-- Desktop sidebar -->
    <div class="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
      <div class="flex flex-col flex-grow bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 pt-5 pb-4 overflow-y-auto">
        <!-- Logo -->
        <div class="flex items-center flex-shrink-0 px-6 mb-8">
          <div class="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center mr-3">
            <Icon name="heroicons:chart-bar-square" class="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 class="text-lg font-bold text-gray-900 dark:text-white">Hydraulic</h1>
            <p class="text-xs text-gray-500 dark:text-gray-400">Диагностика</p>
          </div>
        </div>
        
        <!-- Navigation -->
        <nav class="flex-1 px-3 space-y-1">
          <!-- Privileged navigation -->
          <div v-if="privilegedNavigation.length > 0" class="mb-6">
            <div class="px-3 py-2">
              <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Бизнес-аналитика
              </p>
            </div>
            <NuxtLink
              v-for="item in privilegedNavigation"
              :key="item.name"
              :to="item.href"
              :class="[
                item.current
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg'
                  : 'text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-gray-700 hover:text-blue-600 dark:hover:text-blue-400',
                'group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200'
              ]"
            >
              <Icon :name="item.icon" :class="[
                item.current ? 'text-white' : 'text-gray-400 group-hover:text-blue-500',
                'mr-3 flex-shrink-0 h-5 w-5'
              ]" />
              {{ item.name }}
              <Icon v-if="item.special" name="heroicons:sparkles" class="w-4 h-4 ml-auto text-yellow-400" />
            </NuxtLink>
          </div>
          
          <!-- Standard navigation -->
          <div>
            <div class="px-3 py-2">
              <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Основное
              </p>
            </div>
            <NuxtLink
              v-for="item in navigation"
              :key="item.name"
              :to="item.href"
              :class="[
                item.current
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-gray-700 hover:text-blue-600 dark:hover:text-blue-400',
                'group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200'
              ]"
            >
              <Icon :name="item.icon" :class="[
                item.current ? 'text-white' : 'text-gray-400 group-hover:text-blue-500',
                'mr-3 flex-shrink-0 h-5 w-5'
              ]" />
              {{ item.name }}
            </NuxtLink>
          </div>
        </nav>
        
        <!-- User menu -->
        <div class="flex-shrink-0 px-3">
          <div class="relative" @click.stop>
            <button
              @click="userMenuOpen = !userMenuOpen"
              class="group w-full bg-gray-50 dark:bg-gray-700 rounded-lg p-3 flex items-center text-sm text-left hover:bg-gray-100 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
            >
              <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-3 text-white font-bold text-xs">
                {{ (authStore.user?.first_name?.[0] || 'U').toUpperCase() }}{{ (authStore.user?.last_name?.[0] || '').toUpperCase() }}
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {{ authStore.user?.first_name }} {{ authStore.user?.last_name }}
                </p>
                <p class="text-xs text-gray-500 dark:text-gray-400 truncate">
                  {{ authStore.user?.email }}
                </p>
              </div>
              <Icon name="heroicons:chevron-up-down" class="w-4 h-4 text-gray-400 group-hover:text-gray-500" />
            </button>
            
            <!-- User dropdown -->
            <div
              v-show="userMenuOpen"
              class="absolute bottom-full left-0 right-0 mb-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 py-1"
            >
              <NuxtLink
                v-for="item in userNavigation"
                :key="item.name"
                :to="item.href"
                class="group flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <Icon :name="item.icon" class="w-4 h-4 mr-3 text-gray-400 group-hover:text-gray-500" />
                {{ item.name }}
              </NuxtLink>
              
              <hr class="my-1 border-gray-200 dark:border-gray-600" />
              
              <button
                @click="handleLogout"
                class="group flex items-center w-full px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
              >
                <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-3" />
                Выйти
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Mobile header -->
    <div class="lg:hidden">
      <div class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div class="px-4 sm:px-6">
          <div class="flex items-center justify-between h-16">
            <!-- Mobile logo -->
            <div class="flex items-center">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mr-3">
                <Icon name="heroicons:chart-bar-square" class="w-4 h-4 text-white" />
              </div>
              <h1 class="text-lg font-bold text-gray-900 dark:text-white">Hydraulic</h1>
            </div>
            
            <!-- Mobile menu button -->
            <button
              @click="isMobileMenuOpen = !isMobileMenuOpen"
              class="lg:hidden p-2 text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Icon name="heroicons:bars-3" class="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
      
      <!-- Mobile menu -->
      <div v-show="isMobileMenuOpen" class="lg:hidden bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div class="px-2 pt-2 pb-3 space-y-1">
          <!-- Privileged mobile navigation -->
          <div v-if="privilegedNavigation.length > 0" class="mb-4">
            <div class="px-3 py-2">
              <p class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Бизнес-аналитика
              </p>
            </div>
            <NuxtLink
              v-for="item in privilegedNavigation"
              :key="item.name"
              :to="item.href"
              :class="[
                item.current
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
                  : 'text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-gray-700',
                'group flex items-center px-3 py-2 text-base font-medium rounded-lg'
              ]"
            >
              <Icon :name="item.icon" class="mr-3 flex-shrink-0 h-5 w-5" />
              {{ item.name }}
              <Icon v-if="item.special" name="heroicons:sparkles" class="w-4 h-4 ml-auto text-yellow-400" />
            </NuxtLink>
          </div>
          
          <NuxtLink
            v-for="item in navigation"
            :key="item.name"
            :to="item.href"
            :class="[
              item.current
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-gray-700',
              'group flex items-center px-3 py-2 text-base font-medium rounded-lg'
            ]"
          >
            <Icon :name="item.icon" class="mr-3 flex-shrink-0 h-5 w-5" />
            {{ item.name }}
          </NuxtLink>
        </div>
        
        <!-- Mobile user info -->
        <div class="border-t border-gray-200 dark:border-gray-700 px-4 py-3">
          <div class="flex items-center space-x-3">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
              {{ (authStore.user?.first_name?.[0] || 'U').toUpperCase() }}{{ (authStore.user?.last_name?.[0] || '').toUpperCase() }}
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-base font-medium text-gray-900 dark:text-white">
                {{ authStore.user?.first_name }} {{ authStore.user?.last_name }}
              </p>
              <p class="text-sm text-gray-500 dark:text-gray-400 truncate">
                {{ authStore.user?.email }}
              </p>
            </div>
            <button
              @click="handleLogout"
              class="p-2 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded-lg transition-colors"
            >
              <Icon name="heroicons:arrow-right-on-rectangle" class="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Main content area -->
    <div class="lg:pl-64">
      <!-- Top bar -->
      <div class="sticky top-0 z-10 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div class="px-4 sm:px-6 lg:px-8">
          <div class="flex items-center justify-between h-14">
            <!-- Page title will be inserted here by individual pages -->
            <div class="flex items-center space-x-4">
              <h1 class="text-lg font-semibold text-gray-900 dark:text-white">
                <!-- Dynamic title based on current route -->
                {{ 
                  route.path === '/dashboard' ? 'Обзор систем' :
                  route.path === '/investors' ? 'Бизнес-аналитика' :
                  route.path.startsWith('/equipment') ? 'Оборудование' :
                  route.path === '/diagnostics' ? 'Диагностика' :
                  route.path.startsWith('/reports') ? 'Отчёты' :
                  route.path === '/sensors' ? 'Сенсоры' :
                  route.path === '/chat' ? 'AI-ассистент' :
                  route.path === '/settings' ? 'Настройки' :
                  'Платформа'
                }}
              </h1>
            </div>
            
            <!-- Desktop user menu -->
            <div class="hidden lg:flex items-center space-x-4">
              <!-- Theme toggle -->
              <button 
                @click="$colorMode.preference = $colorMode.value === 'dark' ? 'light' : 'dark'"
                class="p-2 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 rounded-lg transition-colors"
              >
                <Icon 
                  :name="$colorMode.value === 'dark' ? 'heroicons:sun' : 'heroicons:moon'"
                  class="w-5 h-5"
                />
              </button>
              
              <!-- Notifications -->
              <button class="p-2 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 rounded-lg transition-colors relative">
                <Icon name="heroicons:bell" class="w-5 h-5" />
                <span class="absolute top-1 right-1 block h-2 w-2 bg-red-500 rounded-full"></span>
              </button>
              
              <!-- User menu -->
              <div class="relative" @click.stop>
                <button
                  @click="userMenuOpen = !userMenuOpen"
                  class="flex items-center space-x-3 p-2 text-sm rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-xs">
                    {{ (authStore.user?.first_name?.[0] || 'U').toUpperCase() }}{{ (authStore.user?.last_name?.[0] || '').toUpperCase() }}
                  </div>
                  <Icon name="heroicons:chevron-down" class="w-4 h-4 text-gray-400" />
                </button>
                
                <!-- User dropdown -->
                <div
                  v-show="userMenuOpen"
                  class="absolute right-0 bottom-full mb-2 w-56 bg-white dark:bg-gray-800 rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 py-1"
                >
                  <!-- User info -->
                  <div class="px-4 py-3 border-b border-gray-200 dark:border-gray-600">
                    <p class="text-sm font-medium text-gray-900 dark:text-white">
                      {{ authStore.user?.first_name }} {{ authStore.user?.last_name }}
                    </p>
                    <p class="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {{ authStore.user?.email }}
                    </p>
                    <p class="text-xs text-blue-600 dark:text-blue-400 mt-1 capitalize">
                      {{ authStore.user?.role || 'Пользователь' }}
                    </p>
                  </div>
                  
                  <NuxtLink
                    v-for="item in userNavigation"
                    :key="item.name"
                    :to="item.href"
                    class="group flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <Icon :name="item.icon" class="w-4 h-4 mr-3 text-gray-400 group-hover:text-gray-500" />
                    {{ item.name }}
                  </NuxtLink>
                  
                  <hr class="my-1 border-gray-200 dark:border-gray-600" />
                  
                  <button
                    @click="handleLogout"
                    class="group flex items-center w-full px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                  >
                    <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-3" />
                    Выйти
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Page content -->
      <main class="py-8">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <slot />
        </div>
      </main>
    </div>
  </div>
</template>