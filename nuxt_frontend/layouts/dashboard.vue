<script setup lang="ts">
const route = useRoute();

// Safe store initialization
let authStore: any = null;
let colorMode: any = { preference: 'light' };

// Mobile menu state
const isMobileMenuOpen = ref(false)

onMounted(() => {
  try {
    authStore = useAuthStore();
  } catch (e) {
    authStore = {
      user: { name: 'Пользователь', email: 'user@example.com' },
      isAuthenticated: true,
    };
  }

  try {
    colorMode = useColorMode();
  } catch (e) {
    colorMode = { preference: 'light' };
  }
});

// Computed for user
const userName = computed(() => authStore?.user?.name || 'Пользователь');
const userInitials = computed(() => {
  const name = userName.value;
  return name
    .split(' ')
    .map(word => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
});

const toggleTheme = () => {
  if (colorMode?.preference) {
    colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark';
  }
};

const toggleMobileMenu = () => {
  isMobileMenuOpen.value = !isMobileMenuOpen.value
}

const closeMobileMenu = () => {
  isMobileMenuOpen.value = false
}

// Close mobile menu on route change
watch(() => route.path, () => {
  closeMobileMenu()
})

// Breadcrumbs only for deep navigation
const showBreadcrumbs = computed(() => {
  const depth = route.path.split('/').filter(Boolean).length;
  return depth > 1; // Only show if deeper than /dashboard
});

const mapName = (path: string) =>
  ({
    '/dashboard': 'Дашборд',
    '/systems': 'Системы',
    '/diagnostics': 'Диагностика',
    '/reports': 'Отчёты',
    '/chat': 'ИИ Чат',
    '/settings': 'Настройки',
    '/equipments': 'Оборудование',
  })[path] || 'Страница';

const breadcrumbs = computed(() => {
  const parts = route.path.split('/').filter(Boolean);
  const acc: { name: string; href: string }[] = [{ name: 'Главная', href: '/' }];
  let current = '';

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    current += `/${part}`;

    // Handle dynamic routes
    if (part === 'systems') {
      acc.push({ name: 'Системы', href: current });
    } else if (part === 'equipments') {
      acc.push({ name: 'Оборудование', href: current });
    } else if (/^\d+$/.test(part)) {
      // Numeric ID - show as dynamic name
      const prevPart = parts[i - 1];
      if (prevPart === 'systems') {
        acc.push({ name: `Система #${part}`, href: current });
      } else if (prevPart === 'equipments') {
        acc.push({ name: `Оборудование #${part}`, href: current });
      } else {
        acc.push({ name: `#${part}`, href: current });
      }
    } else {
      acc.push({ name: mapName(current), href: current });
    }
  }
  return acc;
});

// Navigation links - removed duplicates
const navigationLinks = [
  { to: '/dashboard', label: 'Обзор', icon: 'i-heroicons-squares-2x2' },
  { to: '/systems', label: 'Системы', icon: 'i-heroicons-server-stack' },
  { to: '/diagnostics', label: 'Диагностика', icon: 'i-heroicons-cpu-chip' },
  { to: '/reports', label: 'Отчёты', icon: 'i-heroicons-document-text' }
]

// Check if link is active
const isActiveLink = (linkPath: string) => {
  if (linkPath === '/dashboard') {
    return route.path === '/dashboard'
  }
  return route.path.startsWith(linkPath)
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Desktop & Mobile Navbar -->
    <nav class="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <!-- Logo section -->
          <div class="flex items-center space-x-3" style="min-width: 220px">
            <NuxtLink to="/" class="flex items-center space-x-2 group" @click="closeMobileMenu">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md">
                <Icon name="i-heroicons-cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <div class="hidden sm:block">
                <span class="text-sm font-bold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors">
                  Гидравлика ИИ
                </span>
                <span class="block text-xs text-gray-500 dark:text-gray-400 leading-tight">
                  Диагностическая платформа
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
                  ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                  : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
              ]"
            >
              <Icon :name="link.icon" class="w-4 h-4" />
              {{ link.label }}
            </NuxtLink>
          </div>

          <!-- Right actions -->
          <div class="flex items-center space-x-3">
            <!-- Search (hidden on small mobile) -->
            <button class="hidden sm:block p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
              <Icon name="i-heroicons-magnifying-glass" class="w-5 h-5" />
            </button>

            <!-- Notifications -->
            <button class="relative p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
              <Icon name="i-heroicons-bell" class="w-5 h-5" />
              <span class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            </button>

            <!-- Theme toggle -->
            <button
              @click="toggleTheme"
              class="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <Icon
                :name="colorMode?.preference === 'dark' ? 'i-heroicons-sun' : 'i-heroicons-moon'"
                class="w-5 h-5"
              />
            </button>

            <!-- User profile -->
            <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md cursor-pointer hover:shadow-lg transition-shadow">
              {{ userInitials }}
            </div>

            <!-- Mobile menu button (shown only on mobile) - FIXED z-index -->
            <button
              @click="toggleMobileMenu"
              class="lg:hidden p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors relative z-50"
              aria-label="Toggle mobile menu"
            >
              <Icon 
                :name="isMobileMenuOpen ? 'i-heroicons-x-mark' : 'i-heroicons-bars-3'"
                class="w-6 h-6" 
              />
            </button>
          </div>
        </div>

        <!-- Mobile Navigation Menu - FIXED positioning -->
        <Transition
          enter-active-class="transition-all duration-200 ease-out"
          enter-from-class="opacity-0 -translate-y-2"
          enter-to-class="opacity-100 translate-y-0"
          leave-active-class="transition-all duration-150 ease-in"
          leave-from-class="opacity-100 translate-y-0"
          leave-to-class="opacity-0 -translate-y-2"
        >
          <div v-if="isMobileMenuOpen" class="absolute top-16 left-0 right-0 lg:hidden border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg z-40">
            <div class="px-4 py-4 space-y-2">
              <!-- Mobile Search -->
              <div class="sm:hidden mb-4">
                <div class="relative">
                  <Icon name="i-heroicons-magnifying-glass" class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search..."
                    class="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
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
                  'flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-colors',
                  isActiveLink(link.to)
                    ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                    : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
                ]"
              >
                <Icon :name="link.icon" class="w-5 h-5" />
                {{ link.label }}
              </NuxtLink>

              <!-- Mobile Footer Links (separate section) -->
              <div class="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4 space-y-2">
                <NuxtLink
                  to="/settings"
                  @click="closeMobileMenu"
                  class="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                >
                  <Icon name="i-heroicons-cog-6-tooth" class="w-5 h-5" />
                  Настройки
                </NuxtLink>
                <NuxtLink
                  to="/chat"
                  @click="closeMobileMenu"
                  class="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                >
                  <Icon name="i-heroicons-chat-bubble-left-right" class="w-5 h-5" />
                  ИИ Помощь
                </NuxtLink>
                <div class="flex items-center gap-3 px-4 py-2 text-xs text-gray-500 dark:text-gray-500">
                  <Icon name="i-heroicons-cpu-chip" class="w-4 h-4" />
                  <span>v{{ $config?.public?.version || '1.0.0' }}</span>
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
      class="bg-white dark:bg-gray-800 border-b border-gray-100 dark:border-gray-700"
    >
      <div class="container mx-auto px-4">
        <nav class="flex items-center space-x-2 text-sm py-3">
          <Icon name="i-heroicons-home" class="w-4 h-4 text-gray-500 dark:text-gray-400" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:underline transition-colors"
              @click="closeMobileMenu"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="font-medium text-gray-900 dark:text-white">
              {{ crumb.name }}
            </span>
            <Icon
              v-if="i < breadcrumbs.length - 1"
              name="i-heroicons-chevron-right"
              class="w-4 h-4 text-gray-400 dark:text-gray-500"
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
    <footer class="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
      <div class="container mx-auto px-4 py-6">
        <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-600 dark:text-gray-400">
          <div class="flex items-center gap-2">
            <Icon name="i-heroicons-cpu-chip" class="w-4 h-4" />
            <span>&copy; 2025 Hydraulic Diagnostic SaaS. All rights reserved.</span>
          </div>
          <div class="flex items-center flex-wrap gap-6">
            <NuxtLink to="/settings" class="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
              Настройки
            </NuxtLink>
            <NuxtLink to="/chat" class="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
              ИИ Помощь
            </NuxtLink>
            <span class="text-xs">
              v{{ $config?.public?.version || '1.0.0' }}
            </span>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<style scoped>
/* Mobile menu proper positioning and z-index */
.mobile-menu {
  z-index: 40;
}
</style>