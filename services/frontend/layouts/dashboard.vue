<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

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
  icon: string
}

interface Breadcrumb {
  name: string
  href: string
}

const route = useRoute()
const { locale, setLocale, t } = useI18n()
const config = useRuntimeConfig()

let authStore: any = null
const isMobileMenuOpen = ref(false)
const isSidebarCollapsed = ref(false)
const showLanguageDropdown = ref(false)
const showUserDropdown = ref(false)
const isOnline = ref(true)
const unreadNotifications = ref(3)

onMounted(() => {
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

  const handleOnline = () => { isOnline.value = true }
  const handleOffline = () => { isOnline.value = false }
  
  window.addEventListener('online', handleOnline)
  window.addEventListener('offline', handleOffline)

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

const availableLocales: LocaleOption[] = [
  { code: 'ru', name: 'Русский', icon: 'circle-flags:ru' },
  { code: 'en', name: 'English', icon: 'circle-flags:us' }
]

const currentLocale = computed<LocaleOption>(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => {
  await setLocale(code as AppLocale)
  showLanguageDropdown.value = false
}

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

watch(() => route.path, () => {
  closeMobileMenu()
})

const emailUser = computed(() => t('landing.footer.contact.emailUser'))
const emailDomain = computed(() => t('landing.footer.contact.emailDomain'))
const emailLabel = computed(() => t('landing.footer.contact.emailLabel'))
const email = computed(() => `${emailUser.value}@${emailDomain.value}`)
const version = computed(() => config?.public?.version || '1.0.0')
</script>

<template>
  <div class="min-h-screen bg-background-primary flex">
    <!-- Desktop Sidebar -->
    <aside
      :class="[
        'hidden lg:flex lg:flex-col card-glass border-r border-steel-700/50 fixed h-screen z-30 transition-all duration-300',
        isSidebarCollapsed ? 'w-20' : 'w-64'
      ]"
    >
      <!-- Sidebar Header -->
      <div class="flex items-center justify-between p-4 border-b border-steel-700/50">
        <UAppLogo 
          v-if="!isSidebarCollapsed"
          :to="'/'"
        />
        
        <button
          @click="toggleSidebar"
          class="btn-icon"
          :title="isSidebarCollapsed ? 'Раскрыть' : t('ui.collapse')"
          :aria-label="isSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
        >
          <Icon
            :name="isSidebarCollapsed ? 'heroicons:chevron-right' : 'heroicons:chevron-left'"
            class="w-5 h-5"
          />
        </button>
      </div>

      <!-- Sidebar Navigation -->
      <nav class="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
        <UAppNavLink
          v-for="link in navigationLinks"
          :key="link.to"
          :to="link.to"
          :label="link.label"
          :icon="link.icon"
          :badge="link.badge"
          :is-active="link.isActive"
          :is-collapsed="isSidebarCollapsed"
        />
      </nav>

      <!-- Sidebar Footer (Online Status) -->
      <div class="p-4 border-t border-steel-700/50">
        <div
          :class="[
            'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm',
            isOnline ? 'bg-success-500/10 text-success-400' : 'bg-red-500/10 text-red-400'
          ]"
        >
          <UStatusDot 
            :status="isOnline ? 'success' : 'error'"
            :animated="isOnline"
          />
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
      <UAppNavbar 
        :user-name="userName"
        :user-email="userEmail"
        :user-initials="userInitials"
        :unread-notifications="unreadNotifications"
        :is-mobile-menu-open="isMobileMenuOpen"
        :show-user-dropdown="showUserDropdown"
        @toggle-mobile-menu="toggleMobileMenu"
        @toggle-user-dropdown="toggleUserDropdown"
        @logout="handleLogout"
      />

      <!-- Breadcrumbs -->
      <UBreadcrumb 
        v-if="showBreadcrumbs"
        :breadcrumbs="breadcrumbs"
      />

      <!-- Main Content -->
      <main class="flex-1 py-8">
        <div class="container-dashboard">
          <slot />
        </div>
      </main>

      <!-- Footer -->
      <footer class="card-glass border-t border-steel-700/50 mt-auto">
        <div class="px-4 sm:px-6 lg:px-8 py-6">
          <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-steel-shine">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
              <span>&copy; 2025 {{ t('app.title') }}. {{ t('footer.copyright') }}.</span>
            </div>
            <div class="flex items-center flex-wrap gap-6">
              <div class="flex items-center gap-1">
                <span>{{ emailLabel }}:</span>
                <a class="hover:text-primary-400 transition-colors" :href="`mailto:${email}`">
                  {{ email }}
                </a>
              </div>
              <NuxtLink to="/chat" class="hover:text-primary-400 transition-colors">
                {{ t('ui.help') }}
              </NuxtLink>
              <span class="text-xs text-steel-400">{{ t('app.version') }} {{ version }}</span>
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
        class="fixed top-0 right-0 h-full w-80 max-w-[90vw] card-glass border-l border-steel-700 shadow-2xl z-50 lg:hidden"
        role="dialog"
        aria-modal="true"
      >
        <div class="flex flex-col h-full">
          <!-- Mobile Header -->
          <div class="flex items-center justify-between p-4 border-b border-steel-700/50">
            <UAppLogo :to="'/'" />
            <button
              @click="closeMobileMenu"
              class="btn-icon"
              aria-label="Close menu"
            >
              <Icon name="heroicons:x-mark" class="w-5 h-5" />
            </button>
          </div>

          <!-- User Section -->
          <div class="p-4 border-b border-steel-700/50">
            <div class="flex items-center space-x-3">
              <div class="w-12 h-12 bg-gradient-to-br from-primary-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                {{ userInitials }}
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-semibold text-white truncate">{{ userName }}</p>
                <p class="text-xs text-steel-shine truncate">{{ userEmail }}</p>
              </div>
            </div>
          </div>

          <!-- Navigation -->
          <div class="flex-1 px-4 py-4 space-y-1 overflow-y-auto scrollbar-thin">
            <UAppNavLink
              v-for="link in navigationLinks"
              :key="link.to"
              :to="link.to"
              :label="link.label"
              :icon="link.icon"
              :badge="link.badge"
              :is-active="link.isActive"
              @click="closeMobileMenu"
            />

            <div class="border-t border-steel-700/50 pt-4 mt-4 space-y-1">
              <UAppNavLink
                to="/profile"
                :label="t('ui.profile')"
                icon="heroicons:user"
                @click="closeMobileMenu"
              />
              <UAppNavLink
                to="/chat"
                :label="t('ui.help')"
                icon="heroicons:question-mark-circle"
                @click="closeMobileMenu"
              />
            </div>
          </div>

          <!-- Mobile Footer -->
          <div class="p-4 border-t border-steel-700/50 space-y-3">
            <!-- Online Status -->
            <div
              :class="[
                'flex items-center space-x-2 px-3 py-2 rounded-lg text-sm',
                isOnline ? 'bg-success-500/10 text-success-400' : 'bg-red-500/10 text-red-400'
              ]"
            >
              <UStatusDot 
                :status="isOnline ? 'success' : 'error'"
                :animated="isOnline"
              />
              <span>{{ isOnline ? t('ui.online') : t('ui.offline') }}</span>
            </div>

            <!-- Language Switcher -->
            <div class="flex items-center justify-between">
              <span class="text-sm text-steel-shine">{{ t('ui.language.switch') }}</span>
              <div class="flex items-center gap-2">
                <button
                  v-for="langOption in availableLocales"
                  :key="langOption.code"
                  @click="switchLanguage(langOption.code)"
                  :class="[
                    'px-3 py-2 rounded-lg text-sm transition-all flex items-center gap-2',
                    currentLocale.code === langOption.code
                      ? 'bg-primary-600/20 text-primary-400 border border-primary-500/50'
                      : 'text-steel-shine hover:bg-steel-800/50'
                  ]"
                >
                  <Icon :name="langOption.icon" class="w-4 h-4" />
                  <span>{{ langOption.code.toUpperCase() }}</span>
                </button>
              </div>
            </div>

            <!-- Logout -->
            <UButton
              variant="destructive"
              class="w-full"
              @click="handleLogout"
            >
              <Icon name="heroicons:arrow-right-on-rectangle" class="w-5 h-5" />
              {{ t('ui.logout') }}
            </UButton>
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
        class="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden"
        @click="closeMobileMenu"
        aria-hidden="true"
      />
    </transition>

    <!-- Modal Portal -->
    <Teleport to="body">
      <div id="modal-portal" />
    </Teleport>
  </div>
</template>

<style scoped>
.container-dashboard {
  @apply px-4 sm:px-6 lg:px-8 max-w-[1600px] mx-auto;
}

/* Custom scrollbar */
aside::-webkit-scrollbar {
  width: 6px;
}

aside::-webkit-scrollbar-track {
  background: transparent;
}

aside::-webkit-scrollbar-thumb {
  background: rgba(76, 89, 111, 0.5);
  border-radius: 3px;
}

aside::-webkit-scrollbar-thumb:hover {
  background: rgba(76, 89, 111, 0.8);
}

#modal-portal {
  position: relative;
  z-index: 60;
}
</style>
