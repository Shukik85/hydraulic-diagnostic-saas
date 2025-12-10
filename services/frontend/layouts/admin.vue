<script setup lang="ts">
import { ref, computed } from 'vue';

const uiStore = useUiStore();
const authStore = useAuthStore();
const { t } = useI18n();

const sidebarOpen = computed(() => uiStore.sidebarOpen);

const navigationItems = [
  {
    name: t('nav.dashboard'),
    href: '/admin/dashboard',
    icon: 'heroicons:chart-bar',
  },
  {
    name: t('nav.tenants'),
    href: '/admin/tenants',
    icon: 'heroicons:building-office',
  },
  {
    name: t('nav.users'),
    href: '/admin/users',
    icon: 'heroicons:users',
  },
  {
    name: t('nav.metrics'),
    href: '/admin/metrics',
    icon: 'heroicons:chart-pie',
  },
  {
    name: t('nav.settings'),
    href: '/admin/settings',
    icon: 'heroicons:cog-6-tooth',
  },
];

const handleLogout = async (): Promise<void> => {
  await authStore.logout();
};
</script>

<template>
  <div class="flex h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Sidebar -->
    <aside
      class="fixed inset-y-0 left-0 z-50 w-64 transform bg-white shadow-lg transition-transform duration-300 dark:bg-gray-800 lg:relative lg:translate-x-0"
      :class="{
        'translate-x-0': sidebarOpen,
        '-translate-x-full': !sidebarOpen,
      }"
    >
      <!-- Logo -->
      <div class="flex h-16 items-center justify-between border-b border-gray-200 px-6 dark:border-gray-700">
        <h1 class="text-xl font-bold text-gray-900 dark:text-white">
          Admin Panel
        </h1>
        <button
          class="lg:hidden"
          @click="uiStore.toggleSidebar()"
          aria-label="Close sidebar"
        >
          <Icon name="heroicons:x-mark" class="h-6 w-6 text-gray-500" />
        </button>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 space-y-1 px-3 py-4">
        <NuxtLink
          v-for="item in navigationItems"
          :key="item.name"
          :to="item.href"
          class="group flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors hover:bg-gray-100 dark:hover:bg-gray-700"
          active-class="bg-primary-50 text-primary-700 dark:bg-primary-900 dark:text-primary-300"
          exact-active-class="bg-primary-100 text-primary-800 dark:bg-primary-800 dark:text-primary-200"
        >
          <Icon :name="item.icon" class="mr-3 h-5 w-5" aria-hidden="true" />
          {{ item.name }}
        </NuxtLink>
      </nav>

      <!-- User Section -->
      <div class="border-t border-gray-200 p-4 dark:border-gray-700">
        <div class="flex items-center">
          <div class="flex h-10 w-10 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900">
            <Icon name="heroicons:user" class="h-6 w-6 text-primary-600 dark:text-primary-300" />
          </div>
          <div class="ml-3 flex-1">
            <p class="text-sm font-medium text-gray-900 dark:text-white">
              {{ authStore.user?.firstName }} {{ authStore.user?.lastName }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              {{ authStore.user?.role }}
            </p>
          </div>
          <button
            @click="handleLogout"
            class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label="Logout"
          >
            <Icon name="heroicons:arrow-right-on-rectangle" class="h-5 w-5" />
          </button>
        </div>
      </div>
    </aside>

    <!-- Mobile overlay -->
    <div
      v-if="sidebarOpen"
      class="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 lg:hidden"
      @click="uiStore.toggleSidebar()"
      aria-hidden="true"
    />

    <!-- Main Content -->
    <div class="flex flex-1 flex-col overflow-hidden">
      <!-- Header -->
      <header class="bg-white shadow-sm dark:bg-gray-800">
        <div class="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
          <button
            class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 lg:hidden"
            @click="uiStore.toggleSidebar()"
            aria-label="Open sidebar"
          >
            <Icon name="heroicons:bars-3" class="h-6 w-6" />
          </button>

          <div class="flex items-center gap-4">
            <!-- Theme toggle -->
            <button
              @click="uiStore.toggleTheme()"
              class="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="Toggle theme"
            >
              <Icon name="heroicons:sun" class="h-5 w-5 dark:hidden" />
              <Icon name="heroicons:moon" class="hidden h-5 w-5 dark:block" />
            </button>

            <!-- Notifications -->
            <button
              class="relative rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="Notifications"
            >
              <Icon name="heroicons:bell" class="h-5 w-5" />
              <span class="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500" />
            </button>
          </div>
        </div>
      </header>

      <!-- Page Content -->
      <main class="flex-1 overflow-auto bg-gray-50 p-4 dark:bg-gray-900 sm:p-6 lg:p-8">
        <slot />
      </main>
    </div>
  </div>
</template>
