<script setup lang="ts">
const colorMode = useColorMode()
const isScrolled = ref(false)

const handleScroll = () => { isScrolled.value = window.scrollY > 60 }

onMounted(() => { if (process.client) window.addEventListener('scroll', handleScroll) })
onUnmounted(() => { if (process.client) window.removeEventListener('scroll', handleScroll) })
</script>

<template>
  <div class="min-h-screen">
    <!-- Navigation - high contrast in all states -->
    <nav :class="[
      'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
      isScrolled 
        ? 'bg-white/96 dark:bg-gray-900/96 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 shadow-sm'
        : 'bg-gradient-to-b from-black/70 to-black/20 backdrop-blur-md'
    ]">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <!-- Logo -->
          <NuxtLink to="/" class="flex items-center space-x-3 group">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 :class="[
                'text-xl font-bold',
                isScrolled ? 'text-gray-900 dark:text-white' : 'text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.5)]'
              ]">Гидравлика ИИ</h1>
              <p :class="[
                'text-xs',
                isScrolled ? 'text-gray-600 dark:text-gray-400' : 'text-white/80 drop-shadow-[0_1px_1px_rgba(0,0,0,0.5)]'
              ]">Диагностическая платформа</p>
            </div>
          </NuxtLink>
          
          <!-- Desktop menu -->
          <div class="hidden md:flex items-center space-x-2">
            <NuxtLink
              v-for="item in [
                { to: '#features', label: 'Возможности' },
                { to: '#benefits', label: 'Преимущества' },
                { to: '/investors', label: 'Для инвесторов' }
              ]"
              :key="item.to"
              :to="item.to"
              :class="[
                'px-3 py-2 rounded-lg text-sm font-medium transition-all',
                isScrolled
                  ? 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                  : 'text-white/90 hover:text-white hover:bg-white/10 drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]'
              ]"
            >
              {{ item.label }}
            </NuxtLink>
          </div>
          
          <!-- Actions -->
          <div class="flex items-center space-x-3">
            <button
              @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'"
              :class="[
                'p-2 rounded-lg transition-colors',
                isScrolled
                  ? 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  : 'text-white/80 hover:text-white hover:bg-white/10 drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]'
              ]"
            >
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
            </button>
            
            <NuxtLink 
              to="/auth/login"
              :class="[
                'px-4 py-2 text-sm font-medium rounded-lg transition-all',
                isScrolled
                  ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  : 'text-white/90 hover:text-white hover:bg-white/10 drop-shadow-[0_1px_1px_rgba(0,0,0,0.35)]'
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
  </div>
</template>
