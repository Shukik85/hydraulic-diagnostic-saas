<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

/**
 * Dashboard Layout Component
 * Enterprise-grade responsive layout с sidebar, header, breadcrumbs
 * Features: i18n, accessibility, mobile-first, real-time indicators
 */

// ==================== TYPES ====================
type AppLocale = 'ru' | 'en'

interface NavigationLink {
  to: string
  label: string
  icon: string
  badge?: number
  isActive?: boolean
}

interface LocaleOption {
  code: AppLocale
  name: string
  flag: string
}

interface Breadcrumb {
  name: string
  href: string
}

interface UserProfile {
  name: string
  email: string
}

// ==================== COMPOSABLES ====================
const route = useRoute()
const { locale, setLocale, t } = useI18n()
const config = useRuntimeConfig()

// ==================== STATE ====================
let authStore: any = null
const isMobileMenuOpen = ref(false)
const isSidebarCollapsed = ref(false)
const showLanguageDropdown = ref(false)
const showUserDropdown = ref(false)
const isOnline = ref(true)
const unreadNotifications = ref(3)

// ==================== LIFECYCLE ====================
onMounted(() => {
  // Auth store (fallback для dev)
  try {
    authStore = useAuthStore()
  } catch {
    authStore = {
      user: {
        name: 'Пользователь',
        email: 'user@example.com'
      }
    }
  }

  // Close dropdowns on outside click
  const handleOutsideClick = (event: Event) => {
    const target = event.target as HTMLElement
    if (!target.closest('.language-dropdown')) {
      showLanguageDropdown.value = false
    }
    if (!target.closest('.user-dropdown')) {
      showUserDropdown.value = false
    }
  }

  document.addEventListener('click', handleOutsideClick)

  // Detect online/offline status
  const handleOnline = () => { isOnline.value = true }
  const handleOffline = () => { isOnline.value = false }
  
  window.addEventListener('online', handleOnline)
  window.addEventListener('offline', handleOffline)

  // Restore sidebar state from localStorage
  const savedSidebarState = localStorage.getItem('sidebarCollapsed')
  if (savedSidebarState !== null) {
    isSidebarCollapsed.value = savedSidebarState === 'true'
  }

  onUnmounted(() => {
    document.removeEventListener('click', handleOutsideClick)
    window.removeEventListener('online', handleOnline)
    window.removeEventListener('offline', handleOffline)
  })
})

// ==================== LOCALES ====================
const availableLocales: LocaleOption[] = [
  { code: 'ru', name: 'Русский', flag: '🇷🇺' },
  { code: 'en', name: 'English', flag: '🇺🇸' }
]

const currentLocale = computed<LocaleOption>(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => {
  await setLocale(code as AppLocale)
  showLanguageDropdown.value = false
}

// ==================== USER ====================
const userName = computed(() => authStore?.user?.name || 'Пользователь')
const userEmail = computed(() => authStore?.user?.email || 'user@example.com')
const userInitials = computed(() =>
  userName.value
    .split(' ')
    .map((w: string) => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)
)

// ==================== NAVIGATION ====================
const navigationLinks = computed<NavigationLink[]>(() => [
  {
    to: '/dashboard',
    label: t('nav.dashboard'),
    icon: 'heroicons:squares-2x2',
    isActive: route.path === '/dashboard'
  },
  {
    to: '/systems',
    label: t('nav.systems'),
    icon: 'heroicons:server-stack',
    isActive: route.path.startsWith('/systems')
  },
  {
    to: '/diagnostics',
    label: t('nav.diagnostics'),
    icon: 'heroicons:cpu-chip',
    badge: unreadNotifications.value,
    isActive: route.path.startsWith('/diagnostics')
  },
  {
    to: '/reports',
    label: t('nav.reports'),
    icon: 'heroicons:document-text',
    isActive: route.path.startsWith('/reports')
  },
  {
    to: '/sensors',
    label: t('nav.sensors'),
    icon: 'heroicons:signal',
    isActive: route.path.startsWith('/sensors')
  },
  {
    to: '/settings',
    label: t('nav.settings'),
    icon: 'heroicons:cog-6-tooth',
    isActive: route.path.startsWith('/settings')
  }
])

const isActiveLink = (linkPath: string): boolean => {
  if (linkPath === '/dashboard') {
    return route.path === '/dashboard'
  }
  return route.path.startsWith(linkPath)
}

// ==================== BREADCRUMBS ====================
const showBreadcrumbs = computed(() => route.path.split('/').filter(Boolean).length > 1)

const mapName = (path: string): string => {
  const nameMap: Record<string, string> = {
    '/dashboard': t('nav.dashboard'),
    '/systems': t('nav.systems'),
    '/diagnostics': t('nav.diagnostics'),
    '/reports': t('nav.reports'),
    '/settings': t('nav.settings'),
    '/sensors': t('nav.sensors'),
    '/equipments': t('nav.equipments')
  }
  return nameMap[path] || t('breadcrumbs.page')
}

const breadcrumbs = computed<Breadcrumb[]>(() => {
  const parts = route.path.split('/').filter(Boolean)
  const acc: Breadcrumb[] = [{ name: t('breadcrumbs.home'), href: '/' }]
  
  let current = ''
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i] || ''
    current += `/${part}`
    
    if (part === 'systems') {
      acc.push({ name: t('nav.systems'), href: current })
    } else if (part === 'equipments') {
      acc.push({ name: t('nav.equipments'), href: current })
    } else if (part && /^\d+$/.test(part)) {
      const prevPart = parts[i - 1]
      if (prevPart === 'systems') {
        acc.push({ name: `${t('breadcrumbs.system')} #${part}`, href: current })
      } else if (prevPart === 'equipments') {
        acc.push({ name: `${t('breadcrumbs.equipment')} #${part}`, href: current })
      } else {
        acc.push({ name: `#${part}`, href: current })
      }
    } else {
      acc.push({ name: mapName(current), href: current })
    }
  }
  
  return acc
})

// ==================== ACTIONS ====================
const toggleMobileMenu = () => {
  isMobileMenuOpen.value = !isMobileMenuOpen.value
}

const toggleSidebar = () => {
  isSidebarCollapsed.value = !isSidebarCollapsed.value
  localStorage.setItem('sidebarCollapsed', String(isSidebarCollapsed.value))
}

const closeMobileMenu = () => {
  isMobileMenuOpen.value = false
  showLanguageDropdown.value = false
  showUserDropdown.value = false
}

const toggleUserDropdown = () => {
  showUserDropdown.value = !showUserDropdown.value
}

const handleLogout = async () => {
  try {
    if (authStore?.logout) {
      await authStore.logout()
    }
  } catch (error) {
    console.error('Logout error:', error)
  }
  await navigateTo('/auth/login')
}

// ==================== WATCHERS ====================
watch(() => route.path, () => {
  closeMobileMenu()
})

// ==================== FOOTER ====================
const emailUser = computed(() => t('landing.footer.contact.emailUser'))
const emailDomain = computed(() => t('landing.footer.contact.emailDomain'))
const emailLabel = computed(() => t('landing.footer.contact.emailLabel'))
const email = computed(() => `${emailUser.value}@${emailDomain.value}`)
const version = computed(() => config?.public?.version || '1.0.0')
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex">
    <!-- Desktop Sidebar -->
    <aside
      :class="[
        'hidden lg:flex lg:flex-col bg-white border-r border-gray-200 fixed h-screen z-30 transition-all duration-300',
        isSidebarCollapsed ? 'w-20' : 'w-64'
      ]"
    >
      <!-- Sidebar Header -->
      <div class="flex items-center justify-between p-4 border-b border-gray-100">
        <NuxtLink
          v-if="!isSidebarCollapsed"
          to="/"
          class="flex items-center space-x-2 group"
        >
          <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
          </div>
          <div>
            <span class="text-sm font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
              {{ t('app.title') }}
            </span>
            <span class="block text-xs text-gray-500 leading-tight">
              {{ t('app.subtitle') }}
            </span>
          </div>
        </NuxtLink>
        
        <button
          @click="toggleSidebar"
          class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          :title="isSidebarCollapsed ? t('ui.expand') : t('ui.collapse')"
          :aria-label="isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
        >
          <Icon
            :name="isSidebarCollapsed ? 'heroicons:chevron-right' : 'heroicons:chevron-left'"
            class="w-5 h-5"
          />
        </button>
      </div>

      <!-- Sidebar Navigation -->
      <nav class="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <NuxtLink
          v-for="link in navigationLinks"
          :key="link.to"
          :to="link.to"
          :class="[
            'flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors text-sm font-medium group relative',
            link.isActive
              ? 'bg-blue-50 text-blue-700'
              : 'text-gray-700 hover:text-blue-700 hover:bg-gray-50'
          ]"
          :title="isSidebarCollapsed ? link.label : undefined"
        >
          <Icon :name="link.icon" class="w-5 h-5 flex-shrink-0" />
          <span v-if="!isSidebarCollapsed" class="flex-1">{{ link.label }}</span>
          
          <!-- Badge -->
          <span
            v-if="link.badge && link.badge > 0 && !isSidebarCollapsed"
            class="px-2 py-0.5 text-xs font-semibold bg-red-500 text-white rounded-full"
          >
            {{ link.badge }}
          </span>
          
          <!-- Collapsed badge indicator -->
          <span
            v-if="link.badge && link.badge > 0 && isSidebarCollapsed"
            class="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"
          ></span>
        </NuxtLink>
      </nav>

      <!-- Sidebar Footer (Online Status) -->
      <div class="p-4 border-t border-gray-100">
        <div
          :class="[
            'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm',
            isOnline ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
          ]"
        >
          <div
            :class="[
              'w-2 h-2 rounded-full flex-shrink-0',
              isOnline ? 'bg-green-500' : 'bg-red-500'
            ]"
          ></div>
          <span v-if="!isSidebarCollapsed">
            {{ isOnline ? t('ui.online') : t('ui.offline') }}
          </span>
        </div>
      </div>
    </aside>

    <!-- Main Content Area -->
    <div
      :class="[
        'flex-1 flex flex-col transition-all duration-300',
        isSidebarCollapsed ? 'lg:ml-20' : 'lg:ml-64'
      ]"
    >
      <!-- Top Navbar -->
      <nav class="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-40">
        <div class="px-4 sm:px-6 lg:px-8">
          <div class="flex items-center justify-between h-16">
            <!-- Mobile Logo -->
            <div class="flex items-center space-x-3 lg:hidden">
              <NuxtLink to="/" class="flex items-center space-x-2">
                <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                  <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
                </div>
                <span class="text-sm font-bold text-gray-900">{{ t('app.title') }}</span>
              </NuxtLink>
            </div>

            <!-- Desktop Actions -->
            <div class="hidden lg:flex items-center flex-1 justify-end space-x-3">
              <!-- Search (placeholder) -->
              <button
                class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                :title="t('ui.search')"
                aria-label="Search"
              >
                <Icon name="heroicons:magnifying-glass" class="w-5 h-5" />
              </button>

              <!-- Help -->
              <NuxtLink
                to="/chat"
                class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                :title="t('ui.help')"
                aria-label="Help"
              >
                <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
              </NuxtLink>

              <!-- Notifications -->
              <button
                class="relative p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
                :title="t('nav.notifications')"
                aria-label="Notifications"
              >
                <Icon name="heroicons:bell" class="w-5 h-5" />
                <span
                  v-if="unreadNotifications > 0"
                  class="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center font-semibold"
                >
                  {{ unreadNotifications }}
                </span>
              </button>

              <!-- Language Dropdown -->
              <div class="relative language-dropdown">
                <button
                  @click="showLanguageDropdown = !showLanguageDropdown"
                  class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors flex items-center gap-1"
                  :aria-label="t('ui.language.switch')"
                  aria-haspopup="true"
                  :aria-expanded="showLanguageDropdown"
                >
                  <Icon name="heroicons:language" class="w-5 h-5" />
                  <span class="text-sm font-medium">{{ currentLocale.code.toUpperCase() }}</span>
                  <Icon
                    name="heroicons:chevron-down"
                    class="w-3 h-3 transition-transform"
                    :class="{ 'rotate-180': showLanguageDropdown }"
                  />
                </button>

                <transition
                  enter-active-class="transition ease-out duration-200"
                  enter-from-class="transform opacity-0 scale-95"
                  enter-to-class="transform opacity-100 scale-100"
                  leave-active-class="transition ease-in duration-150"
                  leave-from-class="transform opacity-100 scale-100"
                  leave-to-class="opacity-0 scale-95"
                >
                  <div
                    v-show="showLanguageDropdown"
                    class="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50"
                    role="menu"
                  >
                    <button
                      v-for="langOption in availableLocales"
                      :key="langOption.code"
                      @click="switchLanguage(langOption.code)"
                      class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3"
                      :class="{ 'bg-blue-50 text-blue-600': currentLocale.code === langOption.code }"
                      role="menuitem"
                    >
                      <span class="text-base">{{ langOption.flag }}</span>
                      <span>{{ langOption.name }}</span>
                      <Icon
                        v-if="currentLocale.code === langOption.code"
                        name="heroicons:check"
                        class="w-4 h-4 ml-auto text-blue-600"
                      />
                    </button>
                  </div>
                </transition>
              </div>

              <!-- User Dropdown -->
              <div class="relative user-dropdown">
                <button
                  @click="toggleUserDropdown"
                  class="flex items-center space-x-2 p-2 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors"
                  aria-label="User menu"
                  aria-haspopup="true"
                  :aria-expanded="showUserDropdown"
                >
                  <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md">
                    {{ userInitials }}
                  </div>
                  <Icon
                    name="heroicons:chevron-down"
                    :class="['w-4 h-4 transition-transform', showUserDropdown ? 'rotate-180' : '']"
                  />
                </button>

                <transition
                  enter-active-class="transition ease-out duration-200"
                  enter-from-class="transform opacity-0 scale-95"
                  enter-to-class="transform opacity-100 scale-100"
                  leave-active-class="transition ease-in duration-150"
                  leave-from-class="transform opacity-100 scale-100"
                  leave-to-class="opacity-0 scale-95"
                >
                  <div
                    v-show="showUserDropdown"
                    class="absolute right-0 mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-200 py-2 z-50"
                    role="menu"
                  >
                    <div class="px-4 py-3 border-b border-gray-100">
                      <div class="flex items-center space-x-3">
                        <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-md">
                          {{ userInitials }}
                        </div>
                        <div class="flex-1 min-w-0">
                          <p class="text-sm font-semibold text-gray-900 truncate">{{ userName }}</p>
                          <p class="text-xs text-gray-600 truncate">{{ userEmail }}</p>
                        </div>
                      </div>
                    </div>

                    <div class="py-1">
                      <NuxtLink
                        to="/profile"
                        class="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        role="menuitem"
                      >
                        <Icon name="heroicons:user" class="w-4 h-4 mr-3" />
                        {{ t('ui.profile') }}
                      </NuxtLink>
                      <NuxtLink
                        to="/settings"
                        class="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        role="menuitem"
                      >
                        <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-3" />
                        {{ t('ui.settings') }}
                      </NuxtLink>
                      <button
                        @click="handleLogout"
                        class="w-full flex items-center px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                        role="menuitem"
                      >
                        <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-3" />
                        {{ t('ui.logout') }}
                      </button>
                    </div>
                  </div>
                </transition>
              </div>
            </div>

            <!-- Mobile Burger -->
            <button
              @click="toggleMobileMenu"
              class="lg:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
              aria-label="Toggle mobile menu"
            >
              <Icon
                :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'"
                class="w-6 h-6"
              />
            </button>
          </div>
        </div>
      </nav>

      <!-- Breadcrumbs -->
      <div v-if="showBreadcrumbs" class="bg-white border-b border-gray-100 sticky top-16 z-30">
        <div class="px-4 sm:px-6 lg:px-8">
          <nav class="flex items-center space-x-2 text-sm py-3" aria-label="Breadcrumb">
            <Icon name="heroicons:home" class="w-4 h-4 text-gray-500" />
            <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
              <NuxtLink
                v-if="i < breadcrumbs.length - 1"
                :to="crumb.href"
                class="text-gray-600 hover:text-blue-600 hover:underline transition-colors"
              >
                {{ crumb.name }}
              </NuxtLink>
              <span v-else class="font-medium text-gray-900">{{ crumb.name }}</span>
              <Icon
                v-if="i < breadcrumbs.length - 1"
                name="heroicons:chevron-right"
                class="w-4 h-4 text-gray-400"
              />
            </template>
          </nav>
        </div>
      </div>

      <!-- Main Content -->
      <main class="flex-1 py-6">
        <div class="px-4 sm:px-6 lg:px-8">
          <slot />
        </div>
      </main>

      <!-- Footer -->
      <footer class="bg-white border-t border-gray-200 mt-auto">
        <div class="px-4 sm:px-6 lg:px-8 py-6">
          <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-600">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
              <span>&copy; 2025 {{ t('app.title') }}. {{ t('footer.copyright') }}.</span>
            </div>
            <div class="flex items-center flex-wrap gap-6">
              <div class="flex items-center gap-1">
                <span>{{ emailLabel }}</span>
                <a class="hover:text-blue-600 transition-colors" :href="`mailto:${email}`">
                  {{ email }}
                </a>
              </div>
              <span>{{ t('landing.footer.contact.phone') }}</span>
              <NuxtLink to="/chat" class="hover:text-blue-600 transition-colors">
                {{ t('landing.footer.contact.chat') }}
              </NuxtLink>
              <span class="text-xs">{{ t('app.version') }} {{ version }}</span>
            </div>
          </div>
        </div>
      </footer>
    </div>

    <!-- Mobile Menu Panel -->
    <transition
      enter-active-class="transition-transform duration-300 ease-out"
      enter-from-class="translate-x-full"
      enter-to-class="translate-x-0"
      leave-active-class="transition-transform duration-200 ease-in"
      leave-from-class="translate-x-0"
      leave-to-class="translate-x-full"
    >
      <div
        v-if="isMobileMenuOpen"
        class="fixed top-0 right-0 h-full w-80 max-w-[90vw] bg-white border-l border-gray-200 shadow-2xl z-50 lg:hidden"
        role="dialog"
        aria-modal="true"
      >
        <div class="flex flex-col h-full">
          <!-- Mobile Header -->
          <div class="flex items-center justify-between p-4 border-b border-gray-200">
            <div class="flex items-center space-x-2">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <span class="text-sm font-bold text-gray-900">{{ t('app.title') }}</span>
            </div>
            <button
              @click="closeMobileMenu"
              class="p-2 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
              aria-label="Close menu"
            >
              <Icon name="heroicons:x-mark" class="w-5 h-5" />
            </button>
          </div>

          <!-- User Section -->
          <div class="p-4 border-b border-gray-200">
            <div class="flex items-center space-x-3">
              <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-md">
                {{ userInitials }}
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-semibold text-gray-900 truncate">{{ userName }}</p>
                <p class="text-xs text-gray-600 truncate">{{ userEmail }}</p>
              </div>
            </div>
          </div>

          <!-- Navigation -->
          <div class="flex-1 px-4 py-4 space-y-1 overflow-y-auto">
            <NuxtLink
              v-for="link in navigationLinks"
              :key="link.to"
              :to="link.to"
              @click="closeMobileMenu"
              :class="[
                'flex items-center justify-between px-4 py-3 rounded-lg transition-colors text-base font-medium',
                link.isActive
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-700 hover:text-blue-700 hover:bg-gray-50'
              ]"
            >
              <div class="flex items-center space-x-3">
                <Icon :name="link.icon" class="w-5 h-5" />
                <span>{{ link.label }}</span>
              </div>
              <span
                v-if="link.badge && link.badge > 0"
                class="px-2 py-0.5 text-xs font-semibold bg-red-500 text-white rounded-full"
              >
                {{ link.badge }}
              </span>
            </NuxtLink>

            <div class="border-t border-gray-200 pt-4 mt-4 space-y-1">
              <NuxtLink
                to="/profile"
                @click="closeMobileMenu"
                class="flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors text-base font-medium text-gray-700 hover:text-blue-700 hover:bg-gray-50"
              >
                <Icon name="heroicons:user" class="w-5 h-5" />
                <span>{{ t('ui.profile') }}</span>
              </NuxtLink>
              <NuxtLink
                to="/chat"
                @click="closeMobileMenu"
                class="flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors text-base font-medium text-gray-700 hover:text-blue-700 hover:bg-gray-50"
              >
                <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
                <span>{{ t('ui.help') }}</span>
              </NuxtLink>
            </div>
          </div>

          <!-- Mobile Footer -->
          <div class="p-4 border-t border-gray-200 space-y-3">
            <!-- Online Status -->
            <div
              :class="[
                'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm',
                isOnline ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
              ]"
            >
              <div
                :class="['w-2 h-2 rounded-full', isOnline ? 'bg-green-500' : 'bg-red-500']"
              ></div>
              <span>{{ isOnline ? t('ui.online') : t('ui.offline') }}</span>
            </div>

            <!-- Language Switcher -->
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600">{{ t('ui.language.switch') }}</span>
              <div class="flex items-center gap-2">
                <button
                  v-for="langOption in availableLocales"
                  :key="langOption.code"
                  @click="switchLanguage(langOption.code)"
                  :class="[
                    'px-3 py-1.5 rounded-md text-sm transition-colors flex items-center gap-2',
                    currentLocale.code === langOption.code
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  ]"
                >
                  <span class="text-base">{{ langOption.flag }}</span>
                  <span>{{ langOption.code.toUpperCase() }}</span>
                </button>
              </div>
            </div>

            <!-- Logout -->
            <button
              @click="handleLogout"
              class="w-full flex items-center justify-center space-x-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4" />
              <span>{{ t('ui.logout') }}</span>
            </button>
          </div>
        </div>
      </div>
    </transition>

    <!-- Mobile Overlay -->
    <transition
      enter-active-class="transition-opacity duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-150"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="isMobileMenuOpen"
        class="fixed inset-0 bg-black/50 z-40 lg:hidden"
        @click="closeMobileMenu"
        aria-hidden="true"
      ></div>
    </transition>

    <!-- Modal Portal -->
    <Teleport to="body">
      <div id="modal-portal"></div>
    </Teleport>
  </div>
</template>

<style scoped>
/* Custom scrollbar for sidebar */
aside::-webkit-scrollbar {
  width: 6px;
}

aside::-webkit-scrollbar-track {
  background: transparent;
}

aside::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

aside::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Modal portal z-index */
#modal-portal {
  position: relative;
  z-index: 60;
}
</style>
