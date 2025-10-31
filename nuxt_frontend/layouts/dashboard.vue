<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue'

type AppLocale = 'ru' | 'en'

const route = useRoute()
const { locale, setLocale, t } = useI18n()

let authStore: any = null
const isMobileMenuOpen = ref(false)
const showLanguageDropdown = ref(false)

onMounted(() => {
  try { authStore = useAuthStore() } catch { authStore = { user: { name: 'Пользователь', email: '' } } }
  document.addEventListener('click', (event: Event) => {
    const target = event.target as HTMLElement
    if (!target.closest('.language-dropdown')) showLanguageDropdown.value = false
  })
})

const availableLocales = [
  { code: 'ru' as AppLocale, name: 'Русский', flag: '🇷🇺' },
  { code: 'en' as AppLocale, name: 'English', flag: '🇺🇸' }
]

const currentLocale = computed(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => { await setLocale(code as AppLocale); showLanguageDropdown.value = false }

const userName = computed(() => authStore?.user?.name || 'Пользователь')
const userInitials = computed(() => userName.value.split(' ').map((w: string) => w[0]).join('').toUpperCase().slice(0, 2))

const toggleMobileMenu = () => { isMobileMenuOpen.value = !isMobileMenuOpen.value }
const closeMobileMenu = () => { isMobileMenuOpen.value = false; showLanguageDropdown.value = false }

watch(() => route.path, () => { closeMobileMenu() })

const showBreadcrumbs = computed(() => route.path.split('/').filter(Boolean).length > 1)

const mapName = (path: string): string => ({
  '/dashboard': t('nav.dashboard'),
  '/systems': t('nav.systems'),
  '/diagnostics': t('nav.diagnostics'),
  '/reports': t('nav.reports'),
  '/chat': t('nav.chat'),
  '/settings': t('nav.settings'),
  '/equipments': t('nav.equipments')
}[path] || t('breadcrumbs.page'))

const breadcrumbs = computed(() => {
  const parts = route.path.split('/').filter(Boolean)
  const acc: { name: string; href: string }[] = [{ name: t('breadcrumbs.home'), href: '/' }]
  let current = ''
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i] || ''
    current += `/${part}`
    if (part === 'systems') acc.push({ name: t('nav.systems'), href: current })
    else if (part === 'equipments') acc.push({ name: t('nav.equipments'), href: current })
    else if (part && /^\d+$/.test(part)) {
      const prevPart = parts[i - 1]
      if (prevPart === 'systems') acc.push({ name: `${t('breadcrumbs.system')} #${part}`, href: current })
      else if (prevPart === 'equipments') acc.push({ name: `${t('breadcrumbs.equipment')} #${part}`, href: current })
      else acc.push({ name: `#${part}`, href: current })
    } else acc.push({ name: mapName(current), href: current })
  }
  return acc
})

const navigationLinks = computed(() => [
  { to: '/dashboard', label: t('nav.dashboard'), icon: 'heroicons:squares-2x2' },
  { to: '/systems', label: t('nav.systems'), icon: 'heroicons:server-stack' },
  { to: '/diagnostics', label: t('nav.diagnostics'), icon: 'heroicons:cpu-chip' },
  { to: '/reports', label: t('nav.reports'), icon: 'heroicons:document-text' }
])

const isActiveLink = (linkPath: string): boolean => linkPath === '/dashboard' ? route.path === '/dashboard' : route.path.startsWith(linkPath)
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <!-- same template as before, using currentLocale safely -->
    <nav class="bg-white border-b border-gray-200 shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center space-x-3" style="min-width: 220px">
            <NuxtLink to="/" class="flex items-center space-x-2 group" @click="closeMobileMenu">
              <div class="w-8 h-8 bg-linear-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <div class="hidden sm:block">
                <span class="text-sm font-bold text-gray-900 group-hover:text-blue-600 transition-colors">{{ t('app.title') }}</span>
                <span class="block text-xs text-gray-500 leading-tight">{{ t('app.subtitle') }}</span>
              </div>
            </NuxtLink>
          </div>

          <div class="hidden lg:flex items-center space-x-6">
            <NuxtLink v-for="link in navigationLinks" :key="link.to" :to="link.to" :class="['px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2', isActiveLink(link.to) ? 'text-blue-700 bg-blue-50' : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50']">
              <Icon :name="link.icon" class="w-4 h-4" />
              {{ link.label }}
            </NuxtLink>
          </div>

          <div class="flex items-center space-x-3">
            <button class="hidden sm:block p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors">
              <Icon name="heroicons:magnifying-glass" class="w-5 h-5" />
            </button>
            <button class="relative p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors">
              <Icon name="heroicons:bell" class="w-5 h-5" />
              <span class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            </button>
            <div class="relative language-dropdown">
              <button @click="showLanguageDropdown = !showLanguageDropdown" class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors flex items-center gap-1" :aria-label="t('ui.language.switch')">
                <Icon name="heroicons:language" class="w-5 h-5" />
                <span class="text-sm font-medium">{{ currentLocale?.code?.toUpperCase() }}</span>
                <Icon name="heroicons:chevron-down" class="w-3 h-3 transition-transform" :class="{ 'rotate-180': showLanguageDropdown }" />
              </button>
              <transition enter-active-class="transition ease-out duration-200" enter-from-class="transform opacity-0 scale-95" enter-to-class="transform opacity-100 scale-100" leave-active-class="transition ease-in duration-150" leave-from-class="transform opacity-100 scale-100" leave-to-class="transform opacity-0 scale-95">
                <div v-show="showLanguageDropdown" class="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50">
                  <button v-for="langOption in availableLocales" :key="langOption.code" @click="switchLanguage(langOption.code)" class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3" :class="{ 'bg-blue-50 text-blue-600': currentLocale?.code === langOption.code }">
                    <span class="text-base">{{ langOption.flag }}</span>
                    <span>{{ langOption.name }}</span>
                    <Icon v-if="currentLocale?.code === langOption.code" name="heroicons:check" class="w-4 h-4 ml-auto text-blue-600" />
                  </button>
                </div>
              </transition>
            </div>
            <div class="w-8 h-8 bg-linear-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md cursor-pointer hover:shadow-lg transition-shadow">{{ userInitials }}</div>
            <button @click="toggleMobileMenu" class="lg:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors relative z-50" aria-label="Toggle mobile menu">
              <Icon :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" class="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
    </nav>

    <Transition enter-active-class="transition-opacity duration-200" enter-from-class="opacity-0" enter-to-class="opacity-100" leave-active-class="transition-opacity duration-150" leave-from-class="opacity-100" leave-to-class="opacity-0">
      <div v-if="isMobileMenuOpen" class="fixed inset-0 bg-black/20 z-30 lg:hidden" @click="closeMobileMenu"></div>
    </Transition>

    <div v-if="showBreadcrumbs" class="bg-white border-b border-gray-100">
      <div class="container mx-auto px-4">
        <nav class="flex items-center space-x-2 text-sm py-3">
          <Icon name="heroicons:home" class="w-4 h-4 text-gray-500" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink v-if="i < breadcrumbs.length - 1" :to="crumb.href" class="text-gray-600 hover:text-blue-600 hover:underline transition-colors" @click="closeMobileMenu">{{ crumb.name }}</NuxtLink>
            <span v-else class="font-medium text-gray-900">{{ crumb.name }}</span>
            <Icon v-if="i < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-400" />
          </template>
        </nav>
      </div>
    </div>

    <main class="py-6"><div class="container mx-auto px-4"><slot /></div></main>
    <footer class="bg-white border-t border-gray-200 mt-16"><div class="container mx-auto px-4 py-6"><div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-600"><div class="flex items-center gap-2"><Icon name="heroicons:cpu-chip" class="w-4 h-4" /><span>&copy; 2025 {{ t('app.title') }}. {{ t('footer.copyright') }}.</span></div><div class="flex items-center flex-wrap gap-6"><NuxtLink to="/settings" class="hover:text-blue-600 transition-colors">{{ t('footer.settings') }}</NuxtLink><NuxtLink to="/chat" class="hover:text-blue-600 transition-colors">{{ t('footer.help') }}</NuxtLink><span class="text-xs">{{ t('app.version') }} {{ $config?.public?.version || '1.0.0' }}</span></div></div></div></footer>
    <Teleport to="body"><div id="modal-portal"></div></Teleport>
  </div>
</template>

<style scoped>
.mobile-menu { z-index: 40; }
</style>
