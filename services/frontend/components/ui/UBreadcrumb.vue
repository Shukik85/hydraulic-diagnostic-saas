<template>
  <nav
    v-if="breadcrumbs.length > 0"
    class="bg-metal-light/10 border-b border-steel-light/10 px-4 sm:px-6 lg:px-8 py-3"
    aria-label="Breadcrumb"
  >
    <ol class="flex items-center space-x-2 text-sm">
      <!-- Home -->
      <li>
        <NuxtLink
          to="/"
          class="flex items-center text-gray-400 hover:text-white transition-colors"
          :aria-label="t('breadcrumb.home')"
        >
          <Icon name="heroicons:home" class="w-4 h-4" />
        </NuxtLink>
      </li>

      <!-- Breadcrumb items -->
      <template v-for="(item, index) in breadcrumbs" :key="item.path">
        <li class="flex items-center">
          <Icon
            name="heroicons:chevron-right"
            class="w-4 h-4 text-gray-600 mx-2"
            aria-hidden="true"
          />
          <NuxtLink
            v-if="index < breadcrumbs.length - 1"
            :to="item.path"
            class="text-gray-400 hover:text-white transition-colors"
          >
            {{ item.label }}
          </NuxtLink>
          <span
            v-else
            class="text-white font-medium"
            aria-current="page"
          >
            {{ item.label }}
          </span>
        </li>
      </template>
    </ol>
  </nav>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const route = useRoute()
const { t } = useI18n()

interface BreadcrumbItem {
  path: string
  label: string
}

// Breadcrumb mapping for translations
const breadcrumbMap: Record<string, string> = {
  dashboard: 'breadcrumb.dashboard',
  systems: 'breadcrumb.systems',
  diagnostics: 'breadcrumb.diagnostics',
  reports: 'breadcrumb.reports',
  chat: 'breadcrumb.chat',
  settings: 'breadcrumb.settings',
  profile: 'breadcrumb.profile',
  security: 'breadcrumb.security',
  sensors: 'breadcrumb.sensors',
  equipments: 'breadcrumb.equipments'
}

// Generate breadcrumbs from route
const breadcrumbs = computed<BreadcrumbItem[]>(() => {
  const pathSegments = route.path.split('/').filter(Boolean)
  const items: BreadcrumbItem[] = []

  let currentPath = ''

  pathSegments.forEach((segment) => {
    currentPath += `/${segment}`

    // Skip dynamic params (e.g., [id])
    if (segment.match(/^\d+$/) || segment.match(/^[a-f0-9-]{36}$/)) {
      // Try to get name from route params or meta
      const label = (route.params.name as string) || (route.meta.title as string) || `#${segment}`
      items.push({
        path: currentPath,
        label
      })
    } else {
      // Use translation key
      const translationKey = breadcrumbMap[segment] || `breadcrumb.${segment}`
      items.push({
        path: currentPath,
        label: t(translationKey)
      })
    }
  })

  return items
})
</script>
