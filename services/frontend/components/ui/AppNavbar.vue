<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

// Props
interface MenuItem {
  to: string
  label: string
  icon?: string
  external?: boolean
}

interface Props {
  items?: MenuItem[]
  showNotifications?: boolean
  showProfile?: boolean
  notificationsCount?: number
}

const props = withDefaults(defineProps<Props>(), {
  items: () => [
    { to: '/dashboard', label: 'dashboard', icon: 'heroicons:squares-2x2' },
    { to: '/systems', label: 'systems', icon: 'heroicons:server-stack' },
    { to: '/diagnostics', label: 'diagnostics', icon: 'heroicons:cpu-chip' },
    { to: '/reports', label: 'reports', icon: 'heroicons:document-text' },
    { to: '/chat', label: 'chat', icon: 'heroicons:chat-bubble-left-ellipsis' },
  ],
  showNotifications: true,
  showProfile: true,
  notificationsCount: 0,
})

// Emits
const emit = defineEmits(['toggle-theme', 'open-notifications', 'open-profile'])

// Composables
const route = useRoute()
const { t } = useI18n()
const colorMode = useColorMode()

// Reactive state
const isMobileMenuOpen = ref(false)
const isProfileMenuOpen = ref(false)

// Safe store initialization
let authStore: any = null

onMounted(() => {
  try {
    authStore = useAuthStore()
  } catch (e) {
    authStore = {
      user: { name: 'User', email: 'user@example.com' },
      isAuthenticated: true,
    }
  }
})

// Computed
const userName = computed(() => authStore?.user?.name || 'User')
const userEmail = computed(() => authStore?.user?.email || 'user@example.com')
const userInitials = computed(() => {
  const name = userName.value
  return name
    .split(' ')
    .map((word: string) => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)
})

// Methods
const toggleTheme = () => {
  if (colorMode?.preference) {
    colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'
  }
  emit('toggle-theme')
}

const openNotifications = () => {
  emit('open-notifications')
}

const toggleProfileMenu = () => {
  isProfileMenuOpen.value = !isProfileMenuOpen.value
}

const handleLogout = () => {
  if (authStore?.logout) {
    authStore.logout()
  }
  navigateTo('/auth/login')
}

// Close mobile menu on route change
watch(
  () => route.path,
  () => {
    isMobileMenuOpen.value = false
  }
)

// Close profile menu when clicking outside
onMounted(() => {
  document.addEventListener('click', (e: Event) => {
    const target = e.target as Element
    if (!target?.closest('.profile-menu')) {
      isProfileMenuOpen.value = false
    }
  })
})
</script>

<template>
  <nav 
    class="fixed top-0 left-0 right-0 z-50 bg-steel-darker border-b border-steel-medium shadow-lg backdrop-blur-sm"
  >
    <div class="container mx-auto flex items-center justify-between h-16 px-4">
      <!-- Logo Section -->
      <slot name="logo">
        <NuxtLink 
          to="/" 
          class="flex items-center space-x-3 group hover:opacity-90 transition-opacity duration-200"
        >
          <div
            class="w-9 h-9 bg-gradient-to-br from-primary-600 to-primary-700 rounded-lg flex items-center justify-center shadow-md shadow-primary-500/20"
          >
            <Icon 
              name="heroicons:cpu-chip" 
              class="w-5 h-5 text-white" 
            />
          </div>
          <div>
            <span
              class="text-lg font-bold text-text-primary group-hover:text-primary-400 transition-colors duration-200 select-none"
            >
              {{ t('app.name', 'Гидравлика ИИ') }}
            </span>
            <span
              class="block text-xs leading-tight text-text-secondary tracking-wide group-hover:text-primary-300 transition-colors"
            >
              {{ t('app.subtitle', 'Диагностическая платформа') }}
            </span>
          </div>
        </NuxtLink>
      </slot>

      <!-- Desktop Navigation -->
      <ul class="hidden lg:flex items-center space-x-1 font-medium">
        <li v-for="item in props.items" :key="item.to">
          <NuxtLink 
            :to="item.to" 
            :target="item.external ? '_blank' : '_self'" 
            :class="[
              'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200',
              route.path === item.to
                ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-md shadow-primary-500/30 font-semibold'
                : 'text-text-secondary hover:text-primary-300 hover:bg-steel-dark',
            ]"
          >
            <Icon v-if="item.icon" :name="item.icon" class="w-4 h-4" />
            <span>{{ t(`nav.${item.label}`) }}</span>
            <Icon 
              v-if="item.external" 
              name="heroicons:arrow-top-right-on-square" 
              class="w-3 h-3 opacity-60" 
            />
          </NuxtLink>
        </li>
      </ul>

      <!-- Desktop Actions -->
      <div class="hidden lg:flex items-center space-x-2">
        <!-- Help (Chat) -->
        <NuxtLink
          to="/chat"
          class="p-2 rounded-lg text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
          :title="t('ui.help')"
        >
          <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
        </NuxtLink>

        <!-- Notifications -->
        <button
          v-if="props.showNotifications"
          @click="openNotifications"
          class="relative p-2 rounded-lg text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
          :title="t('nav.notifications', 'Уведомления')"
        >
          <Icon name="heroicons:bell" class="w-5 h-5" />
          <span
            v-if="props.notificationsCount > 0"
            class="absolute -top-1 -right-1 h-4 min-w-[1rem] px-1 rounded-full bg-error-500 flex items-center justify-center shadow-md shadow-error-500/30"
          >
            <span class="text-xs font-bold text-white">{{ 
              props.notificationsCount > 99 ? '99+' : props.notificationsCount 
            }}</span>
          </span>
        </button>

        <!-- Theme Toggle -->
        <button
          @click="toggleTheme"
          class="p-2 rounded-lg text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
          :title="t('ui.toggleTheme', 'Переключить тему')"
        >
          <Icon 
            :name="colorMode?.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" 
            class="w-5 h-5" 
          />
        </button>

        <!-- User Profile -->
        <div v-if="props.showProfile" class="relative profile-menu">
          <button
            @click="toggleProfileMenu"
            class="flex items-center space-x-2 p-2 rounded-lg text-text-primary hover:bg-steel-dark transition-colors"
          >
            <div
              class="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md shadow-primary-500/20"
            >
              {{ userInitials }}
            </div>
            <Icon
              name="heroicons:chevron-down"
              :class="[
                'w-4 h-4 transition-transform',
                isProfileMenuOpen ? 'rotate-180' : ''
              ]"
            />
          </button>

          <!-- Profile Dropdown -->
          <Transition
            enter-active-class="transition-all duration-200"
            enter-from-class="opacity-0 scale-95"
            enter-to-class="opacity-100 scale-100"
            leave-active-class="transition-all duration-200"
            leave-from-class="opacity-100 scale-100"
            leave-to-class="opacity-0 scale-95"
          >
            <div
              v-if="isProfileMenuOpen"
              class="absolute right-0 top-full mt-2 w-64 bg-steel-darker rounded-lg shadow-xl border border-steel-medium py-2 z-50"
            >
              <div class="px-4 py-3 border-b border-steel-medium">
                <div class="flex items-center space-x-3">
                  <div
                    class="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center text-white font-bold shadow-md shadow-primary-500/20"
                  >
                    {{ userInitials }}
                  </div>
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-semibold text-text-primary truncate">
                      {{ userName }}
                    </p>
                    <p class="text-xs text-text-secondary truncate">{{ userEmail }}</p>
                  </div>
                </div>
              </div>

              <div class="py-1">
                <NuxtLink
                  to="/profile"
                  class="flex items-center px-4 py-2 text-sm text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
                >
                  <Icon name="heroicons:user" class="w-4 h-4 mr-3" />
                  {{ t('ui.profile') }}
                </NuxtLink>
                <NuxtLink
                  to="/settings"
                  class="flex items-center px-4 py-2 text-sm text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
                >
                  <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-3" />
                  {{ t('ui.settings') }}
                </NuxtLink>
                <button
                  @click="handleLogout"
                  class="w-full flex items-center px-4 py-2 text-sm text-error-500 hover:bg-error-500/10 transition-colors"
                >
                  <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-3" />
                  {{ t('ui.logout') }}
                </button>
              </div>
            </div>
          </Transition>
        </div>

        <!-- CTA Button -->
        <slot name="cta">
          <NuxtLink
            to="/dashboard"
            class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg hover:from-primary-700 hover:to-primary-800 shadow-lg shadow-primary-500/30 hover:shadow-xl hover:shadow-primary-500/40 transition-all duration-200"
          >
            {{ t('nav.openDashboard', 'Открыть дашборд') }}
          </NuxtLink>
        </slot>
      </div>

      <!-- Mobile Menu Button -->
      <button
        @click="isMobileMenuOpen = !isMobileMenuOpen"
        class="lg:hidden p-2 text-text-primary rounded-lg hover:bg-steel-dark transition-colors"
      >
        <Icon 
          :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" 
          class="w-6 h-6" 
        />
      </button>
    </div>

    <!-- Mobile Menu -->
    <Transition
      enter-active-class="transition-all duration-200"
      enter-from-class="opacity-0 -translate-y-2"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition-all duration-200"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 -translate-y-2"
    >
      <div 
        v-if="isMobileMenuOpen" 
        class="lg:hidden bg-steel-darker border-t border-steel-medium shadow-lg"
      >
        <div class="px-4 py-4 space-y-1">
          <NuxtLink
            v-for="item in props.items"
            :key="item.to"
            :to="item.to"
            :target="item.external ? '_blank' : '_self'"
            :class="[
              'flex items-center space-x-3 px-3 py-3 rounded-lg transition-colors text-base font-medium',
              route.path === item.to
                ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-md shadow-primary-500/30'
                : 'text-text-secondary hover:text-primary-300 hover:bg-steel-dark',
            ]"
          >
            <Icon v-if="item.icon" :name="item.icon" class="w-5 h-5" />
            <span>{{ t(`nav.${item.label}`) }}</span>
            <Icon
              v-if="item.external"
              name="heroicons:arrow-top-right-on-square"
              class="w-4 h-4 ml-auto opacity-60"
            />
          </NuxtLink>

          <!-- Help (Chat) в мобильном меню -->
          <NuxtLink
            to="/chat"
            class="flex items-center space-x-3 px-3 py-3 rounded-lg transition-colors text-base font-medium text-text-secondary hover:text-primary-300 hover:bg-steel-dark"
          >
            <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
            <span>{{ t('ui.help') }}</span>
          </NuxtLink>

          <div class="border-t border-steel-medium pt-4 mt-4">
            <div class="flex items-center justify-between">
              <button
                @click="toggleTheme"
                class="flex items-center space-x-2 px-3 py-2 rounded-lg text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
              >
                <Icon
                  :name="colorMode?.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'"
                  class="w-4 h-4"
                />
                <span class="text-sm">{{  
                  colorMode?.preference === 'dark' 
                    ? t('ui.lightTheme', 'Светлая') 
                    : t('ui.darkTheme', 'Тёмная')
                }}</span>
              </button>
              <button
                v-if="props.showNotifications"
                @click="openNotifications"
                class="relative p-2 rounded-lg text-text-secondary hover:text-primary-300 hover:bg-steel-dark transition-colors"
              >
                <Icon name="heroicons:bell" class="w-5 h-5" />
                <span
                  v-if="props.notificationsCount > 0"
                  class="absolute -top-1 -right-1 h-4 min-w-[1rem] px-1 rounded-full bg-error-500 flex items-center justify-center shadow-md shadow-error-500/30"
                >
                  <span class="text-xs font-bold text-white">{{ 
                    props.notificationsCount > 99 ? '99+' : props.notificationsCount 
                  }}</span>
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </nav>
</template>