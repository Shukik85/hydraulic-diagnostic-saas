<template>
  <div class="space-y-8">
    <!-- Header ... unchanged ... -->
  </div>
</template>
<script setup lang="ts">
import { useSeoMeta } from '#imports'

definePageMeta({
  title: 'Settings',
  layout: 'dashboard',
  middleware: ['auth']
})

const { t } = useI18n()
const activeTab = ref('profile')

useSeoMeta({
  title: 'Настройки | Hydraulic Diagnostic SaaS',
  description: 'Профиль, безопасность, ключи API, интеграции и права доступа. Всё управление учётной записью в одном месте.',
  ogTitle: 'Settings | Hydraulic Diagnostic SaaS',
  ogDescription: 'Unified user profile, API keys, integrations and access for hydraulic diagnostics SaaS platform.',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

// остальной код без изменений ...
</script>
