<template>
  <div class="h-[calc(100vh-8rem)] flex gap-6">
    <!-- ... template unchanged for brevity ... -->
  </div>
</template>
<script setup lang="ts">
import type { ChatMessage, ChatSession } from '~/types/chat'
import { useSeoMeta } from '#imports'

definePageMeta({
  middleware: ['auth']
})
const { t } = useI18n()

useSeoMeta({
  title: 'Чат консультаций | Hydraulic Diagnostic SaaS',
  description: 'Интерактивный чат с поддержкой ИИ: быстрые консультации по гидравлическим системам, примеры вопросов, поиск по базе знаний.',
  ogTitle: 'Chat | Hydraulic Diagnostic SaaS',
  ogDescription: 'Interactive consultation chat for hydraulic anomaly troubleshooting & expert QA',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})
// ... остальной код без изменений ...
</script>
