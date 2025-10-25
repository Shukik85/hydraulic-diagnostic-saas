<script setup lang="ts">
const authStore = useAuthStore()
const colorMode = useColorMode()
const route = useRoute()

const mapName = (path: string) => ({
  '/dashboard': 'Дашборд',
  '/systems': 'Системы',
  '/diagnostics': 'Диагностика',
  '/reports': 'Отчёты',
  '/chat': 'ИИ Чат',
  '/settings': 'Настройки'
}[path] || 'Страница')

const breadcrumbs = computed(() => {
  const parts = route.path.split('/').filter(Boolean)
  const acc: { name: string, href: string }[] = [{ name: 'Главная', href: '/' }]
  let current = ''
  for (const p of parts) {
    current += `/${p}`
    acc.push({ name: mapName(current), href: current })
  }
  return acc
})
</script>

<template>
  <div class="lg:pl-80">
    <!-- Top bar with breadcrumbs (improved contrast) -->
    <div class="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 dark:border-gray-700 bg-white/96 dark:bg-gray-900/96 backdrop-blur-md px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
      <nav class="flex items-center space-x-2 text-sm" aria-label="Breadcrumb">
        <template v-for="(crumb, index) in breadcrumbs" :key="crumb.href">
          <NuxtLink
            v-if="index < breadcrumbs.length - 1"
            :to="crumb.href"
            class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            {{ crumb.name }}
          </NuxtLink>
          <span v-else class="text-gray-900 dark:text-white font-medium">{{ crumb.name }}</span>
          <Icon v-if="index < breadcrumbs.length - 1" name="heroicons:chevron-right" class="w-4 h-4 text-gray-400 dark:text-gray-500" />
        </template>
      </nav>
    </div>

    <main>
      <slot />
    </main>
  </div>
</template>
