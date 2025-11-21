<script setup lang="ts">
import { ref } from 'vue'

const sidebarOpen = ref(false)
</script>

<template>
  <div class="min-h-screen bg-background-primary">
    <!-- Top Navigation Bar -->
    <UAppNavbar />

    <div class="flex">
      <!-- Sidebar Navigation -->
      <aside
        class="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-background-secondary border-r border-steel-700/50 z-40 transition-transform duration-300"
        :class="sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'"
      >
        <nav class="p-4 space-y-2">
          <NuxtLink
            to="/dashboard"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:home" class="w-5 h-5" />
            <span>Главная</span>
          </NuxtLink>

          <NuxtLink
            to="/systems"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:server-stack" class="w-5 h-5" />
            <span>Системы</span>
          </NuxtLink>

          <NuxtLink
            to="/diagnostics"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:chart-bar" class="w-5 h-5" />
            <span>Диагностика</span>
          </NuxtLink>

          <NuxtLink
            to="/reports"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:document-text" class="w-5 h-5" />
            <span>Отчёты</span>
          </NuxtLink>

          <NuxtLink
            to="/alerts"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:bell" class="w-5 h-5" />
            <span>Уведомления</span>
          </NuxtLink>

          <div class="border-t border-steel-700/50 my-4" />

          <NuxtLink
            to="/settings"
            class="flex items-center gap-3 px-4 py-3 rounded-lg text-steel-shine hover:bg-steel-700/20 hover:text-white transition-colors"
            active-class="bg-primary-600/20 text-primary-400 font-semibold"
          >
            <Icon name="heroicons:cog-6-tooth" class="w-5 h-5" />
            <span>Настройки</span>
          </NuxtLink>
        </nav>
      </aside>

      <!-- Mobile Sidebar Overlay -->
      <div
        v-if="sidebarOpen"
        class="fixed inset-0 bg-black/50 z-30 lg:hidden"
        @click="sidebarOpen = false"
      />

      <!-- Main Content -->
      <main class="flex-1 lg:ml-64 mt-16">
        <div class="container mx-auto px-4 py-8 max-w-7xl">
          <slot />
        </div>
      </main>
    </div>

    <!-- Mobile Menu Toggle -->
    <button
      class="fixed bottom-6 right-6 lg:hidden z-50 w-14 h-14 bg-primary-600 rounded-full shadow-lg flex items-center justify-center text-white hover:bg-primary-700 transition-colors"
      @click="sidebarOpen = !sidebarOpen"
      aria-label="Открыть меню"
    >
      <Icon :name="sidebarOpen ? 'heroicons:x-mark' : 'heroicons:bars-3'" class="w-6 h-6" />
    </button>
  </div>
</template>

<style scoped>
.dashboard-layout {
  @apply min-h-screen;
}
</style>
