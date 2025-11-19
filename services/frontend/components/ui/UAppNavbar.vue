<template>
  <nav class="sticky top-0 z-50 bg-metal-dark border-b border-steel-light/20 backdrop-blur-sm" role="navigation"
    aria-label="Главная навигация">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <!-- Logo -->
        <div class="flex items-center flex-shrink-0">
          <NuxtLink to="/" class="flex items-center gap-3 group" :aria-label="t('nav.home')">
            <div
              class="w-10 h-10 rounded-lg bg-primary-600/20 flex items-center justify-center group-hover:bg-primary-600/30 transition-all">
              <Icon name="heroicons:wrench-screwdriver" class="w-6 h-6 text-primary-400" />
            </div>
            <span class="hidden sm:block text-xl font-bold text-white">
              {{ t('app.title') }}
            </span>
          </NuxtLink>
        </div>

        <!-- Desktop Navigation -->
        <div class="hidden md:flex items-center space-x-1">
          <NuxtLink v-for="item in navigationItems" :key="item.path" :to="item.path"
            class="px-4 py-2 rounded-lg text-sm font-medium transition-all" :class="isActiveRoute(item.path)
              ? 'bg-primary-600/20 text-primary-400'
              : 'text-gray-300 hover:text-white hover:bg-gray-600/10'
              " :aria-current="isActiveRoute(item.path) ? 'page' : undefined">
            <Icon :name="item.icon" class="w-4 h-4 inline mr-2" />
            {{ t(item.labelKey) }}
          </NuxtLink>
        </div>

        <!-- User Menu & Mobile Toggle -->
        <div class="flex items-center gap-2">
          <!-- Language Switcher -->
          <button class="p-2 rounded-lg text-gray-300 hover:text-white hover:bg-gray-600/10 transition-all"
            @click="toggleLanguage" :aria-label="t('nav.switchLanguage')">
            <Icon name="heroicons:language" class="w-5 h-5" />
          </button>

          <!-- User Dropdown (Desktop) -->
          <div class="hidden md:block relative">
            <button class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-600/10 transition-all"
              @click="toggleUserMenu" aria-haspopup="true" :aria-expanded="isUserMenuOpen">
              <div class="w-8 h-8 rounded-full bg-primary-600/20 flex items-center justify-center">
                <Icon name="heroicons:user" class="w-5 h-5 text-primary-400" />
              </div>
              <span class="text-sm font-medium text-white">
                {{ user?.name || 'User' }}
              </span>
              <Icon name="heroicons:chevron-down" class="w-4 h-4 text-gray-400 transition-transform"
                :class="{ 'rotate-180': isUserMenuOpen }" />
            </button>

            <!-- Dropdown Menu -->
            <Transition name="dropdown">
              <div v-if="isUserMenuOpen"
                class="absolute right-0 mt-2 w-56 rounded-lg bg-metal-medium border border-steel-light/20 shadow-lg overflow-hidden"
                role="menu">
                <NuxtLink v-for="item in userMenuItems" :key="item.path" :to="item.path"
                  class="flex items-center gap-3 px-4 py-3 text-sm text-gray-300 hover:bg-gray-600/10 hover:text-white transition-all"
                  role="menuitem" @click="closeUserMenu">
                  <Icon :name="item.icon" class="w-5 h-5" />
                  {{ t(item.labelKey) }}
                </NuxtLink>
                <div class="border-t border-steel-light/20"></div>
                <button
                  class="w-full flex items-center gap-3 px-4 py-3 text-sm text-red-400 hover:bg-red-600/10 transition-all"
                  role="menuitem" @click="handleLogout">
                  <Icon name="heroicons:arrow-right-on-rectangle" class="w-5 h-5" />
                  {{ t('nav.logout') }}
                </button>
              </div>
            </Transition>
          </div>

          <!-- Mobile Menu Toggle -->
          <button class="md:hidden p-2 rounded-lg text-gray-300 hover:text-white hover:bg-gray-600/10 transition-all"
            @click="toggleMobileMenu" :aria-label="t('nav.toggleMenu')" :aria-expanded="isMobileMenuOpen">
            <Icon :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" class="w-6 h-6" />
          </button>
        </div>
      </div>
    </div>

    <!-- Mobile Menu -->
    <Transition name="mobile-menu">
      <div v-if="isMobileMenuOpen" class="md:hidden border-t border-steel-light/20 bg-metal-medium">
        <div class="px-4 py-3 space-y-1">
          <NuxtLink v-for="item in navigationItems" :key="item.path" :to="item.path"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all" :class="isActiveRoute(item.path)
              ? 'bg-primary-600/20 text-primary-400'
              : 'text-gray-300 hover:bg-gray-600/10'
              " @click="closeMobileMenu">
            <Icon :name="item.icon" class="w-5 h-5" />
            {{ t(item.labelKey) }}
          </NuxtLink>

          <div class="border-t border-steel-light/20 my-2"></div>

          <!-- User Menu Mobile -->
          <NuxtLink v-for="item in userMenuItems" :key="item.path" :to="item.path"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-sm text-gray-300 hover:bg-gray-600/10 transition-all"
            @click="closeMobileMenu">
            <Icon :name="item.icon" class="w-5 h-5" />
            {{ t(item.labelKey) }}
          </NuxtLink>

          <button
            class="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm text-red-400 hover:bg-red-600/10 transition-all"
            @click="handleLogout">
            <Icon name="heroicons:arrow-right-on-rectangle" class="w-5 h-5" />
            {{ t('nav.logout') }}
          </button>
        </div>
      </div>
    </Transition>
  </nav>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch, onUnmounted } from 'vue'

const route = useRoute()
const { t, locale } = useI18n()
const router = useRouter()

// State
const isMobileMenuOpen = ref(false)
const isUserMenuOpen = ref(false)

// Mock user data (replace with real auth store)
const user = ref({ name: 'Admin User' })

// Navigation items
const navigationItems = computed(() => [
  {
    path: '/dashboard',
    icon: 'heroicons:squares-2x2',
    labelKey: 'nav.dashboard'
  },
  {
    path: '/systems',
    icon: 'heroicons:server-stack',
    labelKey: 'nav.systems'
  },
  {
    path: '/diagnostics',
    icon: 'heroicons:magnifying-glass',
    labelKey: 'nav.diagnostics'
  },
  {
    path: '/reports',
    icon: 'heroicons:document-text',
    labelKey: 'nav.reports'
  },
  {
    path: '/chat',
    icon: 'heroicons:chat-bubble-left-right',
    labelKey: 'nav.chat'
  }
])

// User menu items
const userMenuItems = computed(() => [
  {
    path: '/settings/profile',
    icon: 'heroicons:user-circle',
    labelKey: 'nav.profile'
  },
  {
    path: '/settings',
    icon: 'heroicons:cog-6-tooth',
    labelKey: 'nav.settings'
  }
])

// Methods
const isActiveRoute = (path: string) => {
  return route.path.startsWith(path)
}

const toggleMobileMenu = () => {
  isMobileMenuOpen.value = !isMobileMenuOpen.value
  isUserMenuOpen.value = false
}

const closeMobileMenu = () => {
  isMobileMenuOpen.value = false
}

const toggleUserMenu = () => {
  isUserMenuOpen.value = !isUserMenuOpen.value
}

const closeUserMenu = () => {
  isUserMenuOpen.value = false
}

const toggleLanguage = () => {
  locale.value = locale.value === 'ru' ? 'en' : 'ru'
}

const handleLogout = async () => {
  // Implement logout logic
  await router.push('/auth/login')
}

// Close menus on route change
watch(() => route.path, () => {
  closeMobileMenu()
  closeUserMenu()
})

// Close dropdowns on outside click
onMounted(() => {
  const handleOutsideClick = (e: MouseEvent) => {
    const target = e.target as HTMLElement
    if (!target.closest('[aria-haspopup]')) {
      closeUserMenu()
    }
  }
  document.addEventListener('click', handleOutsideClick)

  onUnmounted(() => {
    document.removeEventListener('click', handleOutsideClick)
  })
})
</script>

<style scoped>
/* Dropdown transitions */
.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.2s ease;
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* Mobile menu transitions */
.mobile-menu-enter-active,
.mobile-menu-leave-active {
  transition: all 0.3s ease;
}

.mobile-menu-enter-from,
.mobile-menu-leave-to {
  opacity: 0;
  max-height: 0;
}

.mobile-menu-enter-to,
.mobile-menu-leave-from {
  opacity: 1;
  max-height: 500px;
}
</style>
