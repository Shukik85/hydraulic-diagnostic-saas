<template>
  <div class="space-y-8">
    <!-- Header ... unchanged ... -->
  </div>
</template>
<script setup lang="ts">
import { useSeoMeta } from '#imports'

definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

useSeoMeta({
  title: 'Отчёты | Hydraulic Diagnostic SaaS',
  description: 'Генерация, экспорт и просмотр аналитических и технических отчётов с AI-метками и статусом для гидросистем. Быстрый старт и шаблоны.',
  ogTitle: 'Reports | Hydraulic Diagnostic SaaS',
  ogDescription: 'Analytics, technical, compliance and maintenance reports for hydraulic equipment. Export and AI-powered insights.',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})
// ... остальной код без изменений ...
</script>
