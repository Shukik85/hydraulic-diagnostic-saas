<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('diagnostics.title') }}</h1>
        <p class="text-steel-shine mt-2">
          {{ t('diagnostics.subtitle') }}
        </p>
      </div>
      <UButton 
        size="lg"
        @click="showRunModal = true"
      >
        <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
        {{ t('diagnostics.runNew') }}
      </UButton>
    </div>
    <!-- Zero State - показываем только если нет данных -->
    <UZeroState
      v-if="!loading && recentResults.length === 0 && activeSessions.length === 0"
      icon-name="heroicons:document-magnifying-glass"
      :title="t('diagnostics.empty.title')"
      :description="t('diagnostics.empty.description')"
      action-icon="heroicons:play"
      :action-text="t('diagnostics.empty.action')"
      @action="showRunModal = true"
    />
    <!-- Content - показываем только если есть данные -->
    <template v-else>
      <!-- KPI Overview ... (truncated for brevity, unchanged) ... -->
    </template>
    <!-- Run Diagnostic Modal -->
    <URunDiagnosticModal 
      v-model="showRunModal" 
      :loading="isRunning" 
      @submit="startDiagnostic"
    />
    <!-- Results Modal ... (unchanged) ... -->
  </div>
</template>
<script setup lang="ts">
import { useSeoMeta } from '#imports'

definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

useSeoMeta({
  title: 'Диагностика | Hydraulic Diagnostic SaaS',
  description: 'Страница диагностики: запуск новых и просмотр истории гидроанализов с AI-выводом. Быстрый доступ к KPI, статусу и рекомендациям.',
  ogTitle: 'Diagnostics | Hydraulic Diagnostic SaaS',
  ogDescription: 'Diagnostic history, KPI, recommendations and AI insights for hydraulic systems',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

// ...остальной код без изменений...
</script>
