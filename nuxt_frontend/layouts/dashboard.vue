<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue'

const route = useRoute()
const { locale, setLocale, t: $t } = useI18n()

// Safe store initialization
let authStore: any = null

// Mobile menu state
const isMobileMenuOpen = ref(false)

// Language dropdown state
const showLanguageDropdown = ref(false)

onMounted(() => {
  try {
    authStore = useAuthStore()
  } catch (e) {
    authStore = {
      user: { name: 'Пользователь', email: 'user@example.com' },
      isAuthenticated: true,
    }
  }
  
  // Close dropdowns when clicking outside
  document.addEventListener('click', (event: Event) => {
    const target = event.target as HTMLElement
    if (!target.closest('.language-dropdown')) {
      showLanguageDropdown.value = false
    }
  })
})

// Available languages
const availableLocales = [
  { code: 'ru', name: 'Русский', flag: '🇷🇺' },
  { code: 'en', name: 'English', flag: '🇺🇸' }
]

// Current language info
const currentLocale = computed(() => 
  availableLocales.find((l: { code: string; name: string; flag: string }) => l.code === locale.value) || availableLocales[0]
)

// Switch language function
const switchLanguage = async (code: string) => {
  await setLocale(code)
  showLanguageDropdown.value = false
}

// Computed for user
const userName = computed(() => authStore?.user?.name || 'Пользователь')
const userInitials = computed(() => {
  const name = userName.value
  return name
    .split(' ')
    .map((word: string) => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)
})

const toggleMobileMenu = () => {
  isMobileMenuOpen.value = !isMobileMenuOpen.value
}

const closeMobileMenu = () => {
  isMobileMenuOpen.value = false
  showLanguageDropdown.value = false
}

// Close mobile menu on route change
watch(() => route.path, () => {
  closeMobileMenu()
})

// Breadcrumbs only for deep navigation
const showBreadcrumbs = computed(() => {
  const depth = route.path.split('/').filter(Boolean).length
  return depth > 1 // Only show if deeper than /dashboard
})

const mapName = (path: string): string => {
  const mapping: Record<string, string> = {
    '/dashboard': $t('nav.dashboard'),
    '/systems': $t('nav.systems'),
    '/diagnostics': $t('nav.diagnostics'),
    '/reports': $t('nav.reports'),
    '/chat': $t('nav.chat'),
    '/settings': $t('nav.settings'),
    '/equipments': $t('nav.equipments'),
  }
  return mapping[path] || $t('breadcrumbs.page')
}

const breadcrumbs = computed(() => {
  const parts = route.path.split('/').filter(Boolean)
  const acc: { name: string; href: string }[] = [{ name: $t('breadcrumbs.home'), href: '/' }]
  let current = ''

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]
    current += `/${part}`

    // Handle dynamic routes
    if (part === 'systems') {
      acc.push({ name: $t('nav.systems'), href: current })
    } else if (part === 'equipments') {
      acc.push({ name: $t('nav.equipments'), href: current })
    } else if (/^\d+$/.test(part)) {
      // Numeric ID - show as dynamic name
      const prevPart = parts[i - 1]
      if (prevPart === 'systems') {
        acc.push({ name: `${$t('breadcrumbs.system')} #${part}`, href: current })
      } else if (prevPart === 'equipments') {
        acc.push({ name: `${$t('breadcrumbs.equipment')} #${part}`, href: current })
      } else {
        acc.push({ name: `#${part}`, href: current })
      }
    } else {
      acc.push({ name: mapName(current), href: current })
    }
  }
  return acc
})

// Navigation links - with i18n
const navigationLinks = computed(() => [
  { to: '/dashboard', label: $t('nav.dashboard'), icon: 'heroicons:squares-2x2' },
  { to: '/systems', label: $t('nav.systems'), icon: 'heroicons:server-stack' },
  { to: '/diagnostics', label: $t('nav.diagnostics'), icon: 'heroicons:cpu-chip' },
  { to: '/reports', label: $t('nav.reports'), icon: 'heroicons:document-text' }
])

// Check if link is active
const isActiveLink = (linkPath: string): boolean => {
  if (linkPath === '/dashboard') {
    return route.path === '/dashboard'
  }
  return route.path.startsWith(linkPath)
}
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Desktop & Mobile Navbar -->
    <nav class="bg-white border-b border-gray-200 shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <!-- Logo section -->
          <div class="flex items-center space-x-3" style="min-width: 220px">
            <NuxtLink to="/" class="flex items-center space-x-2 group" @click="closeMobileMenu">
              <div class="w-8 h-8 bg-linear-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <div class="hidden sm:block">
                <span class="text-sm font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                  {{ $t('app.title') }}
                </span>
                <span class="block text-xs text-gray-500 leading-tight">
                  {{ $t('app.subtitle') }}
                </span>
              </div>
            </NuxtLink>
          </div>

          <!-- Desktop navigation (hidden on mobile) -->
          <div class="hidden lg:flex items-center space-x-6">
            <NuxtLink
              v-for="link in navigationLinks"
              :key="link.to"
              :to="link.to"
              :class="[
                'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
                isActiveLink(link.to)
                  ? 'text-blue-700 bg-blue-50'
                  : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50',
              ]"
            >
              <Icon :name="link.icon" class="w-4 h-4" />
              {{ link.label }}
            </NuxtLink>
          </div>

          <!-- Right actions -->
          <div class="flex items-center space-x-3">
            <!-- Search (hidden on small mobile) -->
            <button class="hidden sm:block p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors">
              <Icon name="heroicons:magnifying-glass" class="w-5 h-5" />
            </button>

            <!-- Notifications -->
            <button class="relative p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors">
              <Icon name="heroicons:bell" class="w-5 h-5" />
              <span class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            </button>

            <!-- Language Toggle -->
            <div class="relative language-dropdown">
              <button 
                @click="showLanguageDropdown = !showLanguageDropdown"
                class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors flex items-center gap-1"
                :aria-label="$t('ui.language.switch')"
              >
                <Icon name="heroicons:language" class="w-5 h-5" />
                <span class="text-sm font-medium">{{ currentLocale.code.toUpperCase() }}</span>
                <Icon 
                  name="heroicons:chevron-down" 
                  class="w-3 h-3 transition-transform" 
                  :class="{ 'rotate-180': showLanguageDropdown }" 
                />
              </button>
              
              <!-- Language Dropdown -->
              <transition
                enter-active-class="transition ease-out duration-200"
                enter-from-class="transform opacity-0 scale-95"
                enter-to-class="transform opacity-100 scale-100"
                leave-active-class="transition ease-in duration-150"
                leave-from-class="transform opacity-100 scale-100"
                leave-to-class="transform opacity-0 scale-95"
              >
                <div 
                  v-show="showLanguageDropdown" 
                  class="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50"
                >
                  <button
                    v-for="langOption in availableLocales"
                    :key="langOption.code"
                    @click="switchLanguage(langOption.code)"
                    class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3"
                    :class="{ 'bg-blue-50 text-blue-600': currentLocale.code === langOption.code }"
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

            <!-- User profile -->
            <div class="w-8 h-8 bg-linear-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md cursor-pointer hover:shadow-lg transition-shadow">
              {{ userInitials }}
            </div>

            <!-- Mobile menu button (shown only on mobile) -->
            <button
              @click="toggleMobileMenu"
              class="lg:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors relative z-50"
              aria-label="Toggle mobile menu"
            >
              <Icon 
                :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'"
                class="w-6 h-6" 
              />
            </button>
          </div>
        </div>

        <!-- Mobile Navigation Menu -->
        <Transition
          enter-active-class="transition-all duration-200 ease-out"
          enter-from-class="opacity-0 -translate-y-2"
          enter-to-class="opacity-100 translate-y-0"
          leave-active-class="transition-all duration-150 ease-in"
          leave-from-class="opacity-100 translate-y-0"
          leave-to-class="opacity-0 -translate-y-2"
        >
          <div v-if="isMobileMenuOpen" class="absolute top-16 left-0 right-0 lg:hidden border-t border-gray-200 bg-white shadow-lg z-40">
            <div class="px-4 py-4 space-y-2">
              <!-- Mobile Search -->
              <div class="sm:hidden mb-4">
                <div class="relative">
                  <Icon name="heroicons:magnifying-glass" class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    :placeholder="$t('ui.search') + '...'"
                    class="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 bg-gray-50 text-gray-900 placeholder-gray-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                  />
                </div>
              </div>

              <!-- Mobile Navigation Links -->
              <NuxtLink
                v-for="link in navigationLinks"
                :key="link.to"
                :to="link.to"
                @click="closeMobileMenu"
                :class="[
                  'flex items-center gap-3 border-l-4 px-3 py-3 text-base font-medium transition-colors',
                  isActiveLink(link.to)
                    ? 'text-blue-700 bg-blue-50 border-blue-500'
                    : 'border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800',
                ]"
              >
                <Icon :name="link.icon" class="w-5 h-5" />
                {{ link.label }}
              </NuxtLink>

              <!-- Mobile Footer Links -->
              <div class="border-t border-gray-200 pt-4 mt-4 space-y-2">
                <NuxtLink
                  to="/settings"
                  @click="closeMobileMenu"
                  class="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-600 hover:text-blue-600 hover:bg-gray-50 transition-colors"
                >
                  <Icon name="heroicons:cog-6-tooth" class="w-5 h-5" />
                  {{ $t('nav.settings') }}
                </NuxtLink>
                <NuxtLink
                  to="/chat"
                  @click="closeMobileMenu"
                  class="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-600 hover:text-blue-600 hover:bg-gray-50 transition-colors"
                >
                  <Icon name="heroicons:chat-bubble-left-right" class="w-5 h-5" />
                  {{ $t('nav.chat') }}
                </NuxtLink>
                <div class="flex items-center gap-3 px-4 py-2 text-xs text-gray-500">
                  <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
                  <span>{{ $t('app.version') }} {{ $config?.public?.version || '1.0.0' }}</span>
                </div>
              </div>
            </div>
          </div>
        </Transition>
      </div>
    </nav>

    <!-- Mobile menu backdrop -->
    <Transition
      enter-active-class="transition-opacity duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition-opacity duration-150"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div 
        v-if="isMobileMenuOpen" 
        class="fixed inset-0 bg-black/20 z-30 lg:hidden" 
        @click="closeMobileMenu"
      ></div>
    </Transition>

    <!-- Breadcrumbs only for deep navigation -->
    <div
      v-if="showBreadcrumbs"
      class="bg-white border-b border-gray-100"
    >
      <div class="container mx-auto px-4">
        <nav class="flex items-center space-x-2 text-sm py-3">
          <Icon name="heroicons:home" class="w-4 h-4 text-gray-500" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="text-gray-600 hover:text-blue-600 hover:underline transition-colors"
              @click="closeMobileMenu"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="font-medium text-gray-900">
              {{ crumb.name }}
            </span>
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
    <main class="py-6">
      <div class="container mx-auto px-4">
        <slot />
      </div>
    </main>

    <!-- Unified Footer -->
    <footer class="bg-white border-t border-gray-200 mt-16">
      <div class="container mx-auto px-4 py-6">
        <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-600">
          <div class="flex items-center gap-2">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
            <span>&copy; 2025 {{ $t('app.title') }}. {{ $t('footer.copyright') }}.</span>
          </div>
          <div class="flex items-center flex-wrap gap-6">
            <NuxtLink to="/settings" class="hover:text-blue-600 transition-colors">
              {{ $t('footer.settings') }}
            </NuxtLink>
            <NuxtLink to="/chat" class="hover:text-blue-600 transition-colors">
              {{ $t('footer.help') }}
            </NuxtLink>
            <span class="text-xs">
              {{ $t('app.version') }} {{ $config?.public?.version || '1.0.0' }}
            </span>
          </div>
        </div>
      </div>
    </footer>

    <!-- Global Modals Portal -->
    <Teleport to="body">
      <div id="modal-portal"></div>
    </Teleport>
  </div>
</template>

<style scoped>
/* Mobile menu proper positioning and z-index */
.mobile-menu {
  z-index: 40;
}

/* Clean light theme only - no dark mode classes */
</style>