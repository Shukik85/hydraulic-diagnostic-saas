<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';

// Props
interface MenuItem {
  to: string;
  label: string;
  icon?: string;
  external?: boolean;
}

interface Props {
  items?: MenuItem[];
  showNotifications?: boolean;
  showProfile?: boolean;
  notificationsCount?: number;
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
  notificationsCount: 3,
});

// Emits
const emit = defineEmits(['toggle-theme', 'open-notifications', 'open-profile']);

// Reactive state
const isMobileMenuOpen = ref(false);
const isProfileMenuOpen = ref(false);
const route = useRoute();
const { $t } = useI18n();

// Safe store initialization
let authStore: any = null;
let colorMode: any = { preference: 'light' };

onMounted(() => {
  try {
    authStore = useAuthStore();
  } catch (e) {
    authStore = {
      user: { name: 'User', email: 'user@example.com' },
      isAuthenticated: true,
    };
  }

  try {
    colorMode = useColorMode();
  } catch (e) {
    colorMode = { preference: 'light' };
  }
});

// Computed
const userName = computed(() => authStore?.user?.name || 'User');
const userEmail = computed(() => authStore?.user?.email || 'user@example.com');
const userInitials = computed(() => {
  const name = userName.value;
  return name
    .split(' ')
    .map(word => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
});

// Methods
const toggleTheme = () => {
  if (colorMode?.preference) {
    colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark';
  }
  emit('toggle-theme');
};

const openNotifications = () => {
  emit('open-notifications');
};

const toggleProfileMenu = () => {
  isProfileMenuOpen.value = !isProfileMenuOpen.value;
};

const handleLogout = () => {
  if (authStore?.logout) {
    authStore.logout();
  }
  navigateTo('/auth/login');
};

// Close mobile menu on route change
watch(
  () => route.path,
  () => {
    isMobileMenuOpen.value = false;
  }
);

// Close profile menu when clicking outside
onMounted(() => {
  document.addEventListener('click', e => {
    if (!e.target?.closest('.profile-menu')) {
      isProfileMenuOpen.value = false;
    }
  });
});
</script>

<template>
  <nav
    class="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-lg"
  >
    <div class="container mx-auto flex items-center justify-between h-16 px-4">
      <!-- Logo Section -->
      <slot name="logo">
        <NuxtLink
          to="/"
          class="flex items-center space-x-3 group hover:opacity-90 transition-opacity duration-200"
        >
          <div
            class="w-9 h-9 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center shadow-md"
          >
            <Icon
              name="heroicons:cpu-chip"
              class="w-5 h-5 text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]"
            />
          </div>
          <div>
            <span
              class="text-lg font-bold text-gray-900 group-hover:text-blue-700 transition-colors duration-200 drop-shadow-[0_1px_1px_rgba(0,0,0,0.25)] select-none"
            >
              {{ $t('app.name', 'Гидравлика ИИ') }}
            </span>
            <span
              class="block text-xs leading-tight text-gray-600 tracking-wide group-hover:text-blue-600 transition-colors drop-shadow-[0_1px_1px_rgba(0,0,0,0.15)]"
            >
              {{ $t('app.subtitle', 'Диагностическая платформа') }}
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
              'flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 hover:underline',
              route.path === item.to
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md font-semibold'
                : 'text-gray-700 hover:text-blue-700 hover:bg-blue-50',
            ]"
          >
            <Icon v-if="item.icon" :name="item.icon" class="w-4 h-4" />
            <span>{{ $t(`nav.${item.label}`) }}</span>
            <Icon
              v-if="item.external"
              name="heroicons:arrow-top-right-on-square"
              class="w-3 h-3 opacity-60"
            />
          </NuxtLink>
        </li>
      </ul>

      <!-- Desktop Actions -->
      <div class="hidden lg:flex items-center space-x-3">
        <!-- Help (Chat) - заменяет поиск -->
        <NuxtLink
          to="/chat"
          class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          :title="$t('ui.help')"
        >
          <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
        </NuxtLink>

        <!-- Notifications -->
        <button
          v-if="props.showNotifications"
          @click="openNotifications"
          class="relative p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          :title="$t('nav.notifications', 'Уведомления')"
        >
          <Icon name="heroicons:bell" class="w-5 h-5" />
          <span
            v-if="props.notificationsCount > 0"
            class="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 flex items-center justify-center animate-pulse"
          >
            <span class="text-xs font-bold text-white">{{ 
              props.notificationsCount > 99 ? '99+' : props.notificationsCount 
            }}</span>
          </span>
        </button>

        <!-- Theme Toggle -->
        <button
          @click="toggleTheme"
          class="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          :title="$t('ui.toggleTheme', 'Переключить тему')"
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
            class="flex items-center space-x-2 p-2 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors"
          >
            <div
              class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-md"
            >
              {{ userInitials }}
            </div>
            <Icon
              name="heroicons:chevron-down"
              :class="['w-4 h-4 transition-transform', isProfileMenuOpen ? 'rotate-180' : '']"
            />
          </button>

          <!-- Profile Dropdown -->
          <div
            v-if="isProfileMenuOpen"
            class="absolute right-0 top-full mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-200 py-2 z-50"
          >
            <div class="px-4 py-3 border-b border-gray-100">
              <div class="flex items-center space-x-3">
                <div
                  class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-md"
                >
                  {{ userInitials }}
                </div>
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-semibold text-gray-900 truncate">
                    {{ userName }}
                  </p>
                  <p class="text-xs text-gray-600 truncate">{{ userEmail }}</p>
                </div>
              </div>
            </div>

            <div class="py-1">
              <NuxtLink
                to="/profile"
                class="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
              >
                <Icon name="heroicons:user" class="w-4 h-4 mr-3" />
                {{ $t('ui.profile') }}
              </NuxtLink>
              <NuxtLink
                to="/settings"
                class="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
              >
                <Icon name="heroicons:cog-6-tooth" class="w-4 h-4 mr-3" />
                {{ $t('ui.settings') }}
              </NuxtLink>
              <button
                @click="handleLogout"
                class="w-full flex items-center px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
              >
                <Icon name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-3" />
                {{ $t('ui.logout') }}
              </button>
            </div>
          </div>
        </div>

        <!-- CTA Button -->
        <slot name="cta">
          <NuxtLink
            to="/dashboard"
            class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-200"
          >
            {{ $t('nav.openDashboard', 'Открыть дашборд') }}
          </NuxtLink>
        </slot>
      </div>

      <!-- Mobile Menu Button -->
      <button
        @click="isMobileMenuOpen = !isMobileMenuOpen"
        class="lg:hidden p-2 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
      >
        <Icon :name="isMobileMenuOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" class="w-6 h-6" />
      </button>
    </div>

    <!-- Mobile Menu -->
    <div
      v-if="isMobileMenuOpen"
      class="lg:hidden bg-white border-t border-gray-200 shadow-lg"
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
              ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md'
              : 'text-gray-700 hover:text-blue-700 hover:bg-blue-50',
          ]"
        >
          <Icon v-if="item.icon" :name="item.icon" class="w-5 h-5" />
          <span>{{ $t(`nav.${item.label}`) }}</span>
          <Icon
            v-if="item.external"
            name="heroicons:arrow-top-right-on-square"
            class="w-4 h-4 ml-auto opacity-60"
          />
        </NuxtLink>

        <!-- Help (Chat) в мобильном меню -->
        <NuxtLink
          to="/chat"
          class="flex items-center space-x-3 px-3 py-3 rounded-lg transition-colors text-base font-medium text-gray-700 hover:text-blue-700 hover:bg-blue-50"
        >
          <Icon name="heroicons:question-mark-circle" class="w-5 h-5" />
          <span>{{ $t('ui.help') }}</span>
        </NuxtLink>

        <div class="border-t border-gray-200 pt-4 mt-4">
          <div class="flex items-center justify-between">
            <button
              @click="toggleTheme"
              class="flex items-center space-x-2 px-3 py-2 rounded-lg text-gray-700 hover:bg-gray-100 transition-colors"
            >
              <Icon
                :name="colorMode?.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'"
                class="w-4 h-4"
              />
              <span class="text-sm">{{ 
                colorMode?.preference === 'dark' ? $t('ui.lightTheme', 'Светлая') : $t('ui.darkTheme', 'Тёмная')
              }}</span>
            </button>
            <button
              v-if="props.showNotifications"
              @click="openNotifications"
              class="relative p-2 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
            >
              <Icon name="heroicons:bell" class="w-5 h-5" />
              <span
                v-if="props.notificationsCount > 0"
                class="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 flex items-center justify-center animate-pulse"
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
  </nav>
</template>

<style scoped>
/* Additional styles if needed */
.profile-menu {
  /* Ensure click outside detection works */
}
</style>