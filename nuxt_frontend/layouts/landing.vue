<script setup lang="ts">
const isDark = ref(false)
const isHydrated = ref(false)
const isScrolled = ref(false)

// Handle scroll for navbar style changes
const handleScroll = () => {
  isScrolled.value = window.scrollY > 50
}

// Safe theme toggle
const toggleTheme = () => {
  if (!isHydrated.value) return
  isDark.value = !isDark.value
  if (process.client) {
    document.documentElement.classList.toggle('dark', isDark.value)
    localStorage.setItem('color-mode', isDark.value ? 'dark' : 'light')
  }
}

onMounted(() => {
  // Initialize theme safely
  const stored = localStorage.getItem('color-mode')
  isDark.value = stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', isDark.value)
  isHydrated.value = true
  
  // Add scroll listener
  if (process.client) {
    window.addEventListener('scroll', handleScroll)
  }
})

onUnmounted(() => {
  if (process.client) {
    window.removeEventListener('scroll', handleScroll)
  }
})
</script>

<template>
  <div class="min-h-screen">
    <!-- Fixed navbar with proper opacity and no flicker -->
    <nav :class="[
      'sticky top-0 z-50 transition-all duration-300',
      isScrolled 
        ? 'bg-white/95 dark:bg-gray-900/95 backdrop-blur-md shadow-lg border-b border-gray-200/50 dark:border-gray-700/50'
        : 'bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm'
    ]">
      <div class="container mx-auto px-6">
        <div class="flex items-center justify-between h-16">
          <!-- Logo with breathing room -->
          <NuxtLink to="/" class="flex items-center space-x-3 mr-8 group">
            <div class="w-9 h-9 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <span :class="[
                'text-lg font-bold group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors',
                isScrolled
                  ? 'text-gray-900 dark:text-white'
                  : 'text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]'
              ]">
                Гидравлика ИИ
              </span>
              <span :class="[
                'block text-xs leading-tight',
                isScrolled
                  ? 'text-gray-600 dark:text-gray-400'
                  : 'text-blue-100/90 drop-shadow-[0_1px_1px_rgba(0,0,0,0.2)]'
              ]">
                Диагностическая платформа
              </span>
            </div>
          </NuxtLink>

          <!-- Simple navigation -->
          <div class="hidden md:flex items-center space-x-8">
            <a href="#features" :class="[
              'text-sm font-medium transition-colors hover:underline',
              isScrolled
                ? 'text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]'
            ]">
              Возможности
            </a>
            <a href="#benefits" :class="[
              'text-sm font-medium transition-colors hover:underline',
              isScrolled
                ? 'text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]'
            ]">
              Преимущества
            </a>
            <NuxtLink to="/investors" :class="[
              'text-sm font-medium transition-colors hover:underline',
              isScrolled
                ? 'text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]'
            ]">
              Для инвесторов
            </NuxtLink>
          </div>

          <!-- Actions -->
          <div class="flex items-center space-x-3">
            <!-- Theme toggle with hydration fix -->
            <button 
              @click="toggleTheme" 
              :disabled="!isHydrated"
              :class="[
                'p-2 rounded-lg transition-colors disabled:opacity-50',
                isScrolled
                  ? 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800'
                  : 'text-white/80 hover:text-white hover:bg-white/10'
              ]"
              title="Переключить тему"
            >
              <ClientOnly>
                <Icon :name="isDark ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
                <template #fallback>
                  <Icon name="heroicons:moon" class="w-5 h-5 opacity-60" />
                </template>
              </ClientOnly>
            </button>
            
            <!-- Auth actions -->
            <NuxtLink 
              to="/auth/login" 
              :class="[
                'hidden sm:inline-flex px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                isScrolled
                  ? 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                  : 'text-white/90 hover:text-white hover:bg-white/10'
              ]"
            >
              Войти
            </NuxtLink>
            
            <NuxtLink 
              to="/dashboard" 
              class="px-6 py-2.5 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all duration-200"
            >
              Открыть дашборд
            </NuxtLink>
          </div>

          <!-- Mobile menu button -->
          <button class="md:hidden p-2 rounded-lg transition-colors" :class="[
            isScrolled
              ? 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              : 'text-white/80 hover:bg-white/10'
          ]">
            <Icon name="heroicons:bars-3" class="w-6 h-6" />
          </button>
        </div>
      </div>
    </nav>
    
    <!-- Main content -->
    <main>
      <slot />
    </main>
    
    <!-- Footer remains as before -->
    <footer class="bg-gray-900 dark:bg-black text-white">
      <div class="container mx-auto px-4 py-12">
        <div class="grid md:grid-cols-4 gap-8">
          <div class="md:col-span-2">
            <div class="flex items-center space-x-3 mb-4">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
              </div>
              <h3 class="text-lg font-bold">Гидравлика ИИ</h3>
            </div>
            <p class="text-gray-300 mb-6 leading-relaxed">
              Передовая платформа диагностики и мониторинга промышленных гидравлических систем
              с использованием искусственного интеллекта и машинного обучения.
            </p>
          </div>
          
          <div>
            <h4 class="text-sm font-semibold uppercase tracking-wider mb-4">Платформа</h4>
            <ul class="space-y-2 text-gray-300">
              <li><a href="#features" class="hover:text-white transition-colors">Возможности</a></li>
              <li><NuxtLink to="/auth/register" class="hover:text-white transition-colors">Регистрация</NuxtLink></li>
              <li><NuxtLink to="/auth/login" class="hover:text-white transition-colors">Вход в систему</NuxtLink></li>
              <li><NuxtLink to="/investors" class="hover:text-white transition-colors">Демо-версия</NuxtLink></li>
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
/* Ensure stable rendering and no flicker */
nav {
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  transform: translateZ(0);
}

/* Override any conflicting transparent styles */
:deep(.navbar),
:deep(nav[class*="transparent"]) {
  background: rgba(255, 255, 255, 0.9) !important;
}

:deep(.dark .navbar),
:deep(.dark nav[class*="transparent"]) {
  background: rgba(17, 24, 39, 0.9) !important;
}
</style>