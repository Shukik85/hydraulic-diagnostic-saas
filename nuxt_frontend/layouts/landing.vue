<script setup lang="ts">
// Landing layout with enhanced navigation
const colorMode = useColorMode()
const isScrolled = ref(false)

// Handle scroll for navbar transparency
const handleScroll = () => {
  isScrolled.value = window.scrollY > 50
}

onMounted(() => {
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
    <!-- Navigation -->
    <nav :class="[
      'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
      isScrolled 
        ? 'bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm shadow-lg border-b border-gray-200 dark:border-gray-700'
        : 'bg-transparent'
    ]">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <!-- Logo -->
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-80 transition-opacity">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 class="text-xl font-bold" :class="[
                isScrolled 
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent'
                  : 'text-white'
              ]">
                Гидравлика ИИ
              </h1>
              <p class="text-xs" :class="[
                isScrolled
                  ? 'text-gray-600 dark:text-gray-400'
                  : 'text-blue-100'
              ]">
                Диагностическая платформа
              </p>
            </div>
          </NuxtLink>
          
          <!-- Navigation menu -->
          <div class="hidden md:flex items-center space-x-6">
            <a href="#features" :class="[
              'text-sm font-medium transition-colors',
              isScrolled
                ? 'text-gray-700 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white'
            ]">
              Возможности
            </a>
            <a href="#benefits" :class="[
              'text-sm font-medium transition-colors',
              isScrolled
                ? 'text-gray-700 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white'
            ]">
              Преимущества
            </a>
            <NuxtLink to="/investors" :class="[
              'text-sm font-medium transition-colors',
              isScrolled
                ? 'text-gray-700 hover:text-blue-600 dark:text-gray-300 dark:hover:text-blue-400'
                : 'text-white/90 hover:text-white'
            ]">
              Для инвесторов
            </NuxtLink>
          </div>
          
          <!-- Actions -->
          <div class="flex items-center space-x-3">
            <button
              @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'"
              :class="[
                'p-2 rounded-lg transition-colors',
                isScrolled
                  ? 'text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                  : 'text-white/80 hover:text-white hover:bg-white/10'
              ]"
            >
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
            </button>
            
            <NuxtLink 
              to="/auth/login"
              :class="[
                'px-4 py-2 text-sm font-medium rounded-lg transition-all',
                isScrolled
                  ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                  : 'text-white/90 hover:text-white hover:bg-white/10'
              ]"
            >
              Войти
            </NuxtLink>
            
            <NuxtLink 
              to="/auth/register"
              class="px-4 py-2 text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
            >
              Начать бесплатно
            </NuxtLink>
          </div>
        </div>
      </div>
    </nav>
    
    <!-- Page content -->
    <div class="pt-16">
      <slot />
    </div>
    
    <!-- Footer -->
    <footer class="bg-gray-900 dark:bg-black text-white">
      <div class="container mx-auto px-4 py-12">
        <div class="grid md:grid-cols-4 gap-8">
          <!-- Company info -->
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
            <div class="flex space-x-4">
              <a href="#" class="text-gray-400 hover:text-white transition-colors">
                <Icon name="heroicons:envelope" class="w-5 h-5" />
              </a>
              <a href="#" class="text-gray-400 hover:text-white transition-colors">
                <Icon name="heroicons:phone" class="w-5 h-5" />
              </a>
            </div>
          </div>
          
          <!-- Quick Links -->
          <div>
            <h4 class="text-sm font-semibold uppercase tracking-wider mb-4">Платформа</h4>
            <ul class="space-y-2 text-gray-300">
              <li><a href="#features" class="hover:text-white transition-colors">Возможности</a></li>
              <li><NuxtLink to="/auth/register" class="hover:text-white transition-colors">Регистрация</NuxtLink></li>
              <li><NuxtLink to="/auth/login" class="hover:text-white transition-colors">Вход в систему</NuxtLink></li>
              <li><NuxtLink to="/investors" class="hover:text-white transition-colors">Демо-версия</NuxtLink></li>
            </ul>
          </div>
          
          <!-- Support -->
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