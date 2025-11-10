<script setup lang="ts">
const isDark = ref(false);
const isHydrated = ref(false);

const toggleTheme = () => {
  if (!isHydrated.value) return;
  isDark.value = !isDark.value;
  if (process.client) {
    document.documentElement.classList.toggle('dark', isDark.value);
    localStorage.setItem('color-mode', isDark.value ? 'dark' : 'light');
  }
};

onMounted(() => {
  const stored = localStorage.getItem('color-mode');
  isDark.value =
    stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.classList.toggle('dark', isDark.value);
  isHydrated.value = true;
});
</script>

<template>
  <div class="min-h-screen">
    <!-- Навбар c лёгкой уместной прозрачностью и без эффектов/анимаций -->
    <nav
      class="sticky top-0 z-50 bg-white/90 dark:bg-gray-900/90 border-b border-gray-200/70 dark:border-gray-800/70"
    >
      <div class="container mx-auto px-6">
        <div class="flex items-center justify-between h-16">
          <!-- Логотип -->
          <NuxtLink to="/" class="flex items-center space-x-3 mr-8">
            <div
              class="w-9 h-9 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center"
            >
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <span class="text-lg font-bold text-gray-900 dark:text-white">Гидравлика ИИ</span>
              <span class="block text-xs text-gray-600 dark:text-gray-400 leading-tight"
                >Диагностическая платформа</span
              >
            </div>
          </NuxtLink>

          <!-- Публичные ссылки с единым стилем -->
          <div class="hidden md:flex items-center space-x-8">
            <a
              href="#features"
              class="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
              >Возможности</a
            >
            <a
              href="#benefits"
              class="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
              >Преимущества</a
            >
            <NuxtLink
              to="/investors"
              class="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
              >Для инвесторов</NuxtLink
            >
          </div>

          <!-- Действия -->
          <div class="flex items-center space-x-3">
            <button
              @click="toggleTheme"
              :disabled="!isHydrated"
              class="p-2 rounded-lg text-gray-600 hover:text-blue-600 transition-colors disabled:opacity-50"
              title="Переключить тему"
            >
              <ClientOnly>
                <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
                <template #fallback>
                  <Icon name="heroicons:moon" class="w-5 h-5" />
                </template>
              </ClientOnly>
            </button>
            <NuxtLink
              to="/auth/login"
              class="px-4 py-2 text-sm font-medium rounded-lg text-gray-600 hover:text-blue-600 transition-colors"
              >Войти</NuxtLink
            >
            <NuxtLink
              to="/dashboard"
              class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-colors"
              >Открыть дашборд</NuxtLink
            >
          </div>

          <!-- Mobile button -->
          <button class="md:hidden p-2 text-gray-600 hover:text-blue-600 transition-colors">
            <Icon name="heroicons:bars-3" class="w-6 h-6" />
          </button>
        </div>
      </div>
    </nav>

    <main>
      <slot />
    </main>

    <footer class="bg-gray-900 dark:bg-black text-white">
      <div class="container mx-auto px-4 py-12">
        <div class="grid md:grid-cols-4 gap-8">
          <div class="md:col-span-2">
            <div class="flex items-center space-x-3 mb-4">
              <div
                class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center"
              >
                <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <h3 class="text-lg font-bold">Гидравлика ИИ</h3>
            </div>
            <p class="text-gray-300 mb-6 leading-relaxed">
              Передовая платформа диагностики и мониторинга промышленных гидравлических систем с
              использованием искусственного интеллекта.
            </p>
          </div>

          <div>
            <h4 class="text-sm font-semibold uppercase tracking-wider mb-4">Платформа</h4>
            <ul class="space-y-2 text-gray-300">
              <li>
                <a href="#features" class="hover:text-white transition-colors">Возможности</a>
              </li>
              <li>
                <NuxtLink to="/auth/register" class="hover:text-white transition-colors"
                  >Регистрация</NuxtLink
                >
              </li>
              <li>
                <NuxtLink to="/auth/login" class="hover:text-white transition-colors"
                  >Вход в систему</NuxtLink
                >
              </li>
              <li>
                <NuxtLink to="/investors" class="hover:text-white transition-colors"
                  >Демо-версия</NuxtLink
                >
              </li>
            </ul>
          </div>

          <div>
            <h4 class="text-sm font-semibold uppercase tracking-wider mb-4">Поддержка</h4>
            <ul class="space-y-2 text-gray-300">
              <li><a href="#" class="hover:text-white transition-colors">Документация</a></li>
              <li><a href="#" class="hover:text-white transition-colors">Техподдержка</a></li>
              <li><a href="#" class="hover:text-white transition-colors">Обратная связь</a></li>
              <li><a href="#" class="hover:text-white transition-colors">Статус системы</a></li>
            </ul>
          </div>
        </div>

        <div class="border-t border-gray-800 mt-12 pt-8 text-center">
          <p class="text-gray-400 text-sm">
            © {{ new Date().getFullYear() }} Гидравлика ИИ. Все права защищены.
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<style scoped>
/* Без теней, без backdrop, без анимаций; лёгкая прозрачность через /90 */
nav {
  box-shadow: none;
}
</style>
