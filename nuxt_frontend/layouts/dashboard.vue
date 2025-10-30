<script setup lang="ts">
const route = useRoute();

// Safe store initialization
let authStore: any = null;
let colorMode: any = { preference: 'light' };

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
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Unified Dashboard Navbar -->
    <nav class="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div class="container mx-auto flex items-center justify-between h-16 px-4">
        <!-- Fixed width logo section -->
        <div class="flex items-center space-x-3" style="min-width: 220px">
          <NuxtLink to="/" class="flex items-center space-x-2 group">
            <div
              class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md"
            >
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
            </div>
            <div>
              <span
                class="text-sm font-bold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors"
              >
                Гидравлика ИИ
              </span>
              <span class="block text-xs text-gray-500 dark:text-gray-400 leading-tight">
                Диагностическая платформа
              </span>
            </div>
          </NuxtLink>
        </div>

        <!-- Core navigation - unified across all pages -->
        <div class="hidden lg:flex items-center space-x-6">
          <NuxtLink
            to="/dashboard"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
              route.path === '/dashboard'
                ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
            ]"
          >
            <Icon name="heroicons:squares-2x2" class="w-4 h-4" />
            Обзор
          </NuxtLink>
          <NuxtLink
            to="/systems"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
              route.path.startsWith('/systems')
                ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
            ]"
          >
            <Icon name="heroicons:server-stack" class="w-4 h-4" />
            Системы
          </NuxtLink>
          <NuxtLink
            to="/diagnostics"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
              route.path === '/diagnostics'
                ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
            ]"
          >
            <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
            Диагностика
          </NuxtLink>
          <NuxtLink
            to="/reports"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2',
              route.path === '/reports'
                ? 'text-blue-700 bg-blue-50 dark:text-blue-300 dark:bg-blue-900/30'
                : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800',
            ]"
          >
            <Icon name="heroicons:document-text" class="w-4 h-4" />
            Отчёты
          </NuxtLink>
        </div>

        <!-- Right actions - consistent across all pages -->
        <div class="flex items-center space-x-3">
          <!-- Search -->
          <button
            class="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <Icon name="heroicons:magnifying-glass" class="w-5 h-5" />
          </button>

          <!-- Notifications -->
          <button
            class="relative p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <Icon name="heroicons:bell" class="w-5 h-5" />
            <span
              class="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"
            ></span>
          </button>

          <!-- Theme toggle -->
          <button
            @click="toggleTheme"
            class="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <Icon
              :name="colorMode?.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'"
              class="w-5 h-5"
            />
          </button>

          <!-- User profile - consistent height with other buttons -->
          <div
            class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md cursor-pointer hover:shadow-lg transition-shadow"
          >
            {{ userInitials }}
          </div>

          <!-- Green button REMOVED per requirements -->
        </div>
      </div>
    </nav>

    <!-- Breadcrumbs only for deep navigation -->
    <div
      v-if="showBreadcrumbs"
      class="bg-white dark:bg-gray-800 border-b border-gray-100 dark:border-gray-700"
    >
      <div class="container mx-auto px-4">
        <nav class="flex items-center space-x-2 text-sm py-3">
          <Icon name="heroicons:home" class="w-4 h-4 text-gray-500 dark:text-gray-400" />
          <template v-for="(crumb, i) in breadcrumbs" :key="crumb.href">
            <NuxtLink
              v-if="i < breadcrumbs.length - 1"
              :to="crumb.href"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:underline transition-colors"
            >
              {{ crumb.name }}
            </NuxtLink>
            <span v-else class="font-medium text-gray-900 dark:text-white">
              {{ crumb.name }}
            </span>
            <Icon
              v-if="i < breadcrumbs.length - 1"
              name="heroicons:chevron-right"
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

    <!-- Unified Footer - simple and consistent -->
    <footer class="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
      <div class="container mx-auto px-4 py-6">
        <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
          <div class="flex items-center gap-2">
            <Icon name="heroicons:cpu-chip" class="w-4 h-4" />
            <span>&copy; 2025 Hydraulic Diagnostic SaaS. All rights reserved.</span>
          </div>
          <div class="flex items-center gap-6">
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
