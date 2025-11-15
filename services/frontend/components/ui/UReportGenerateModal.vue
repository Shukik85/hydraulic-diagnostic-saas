<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'

// emulates async/UX - replace with store or composable in production
const loading = ref(false)
const modelValue = ref(true)
const { t } = useI18n()

const form = ref({
  template: 'executive',
  range: 'last_7d',
  locale: 'ru-RU',
  title: ''
})

function handleCancel() {
  modelValue.value = false
}
function handleSubmit() {
  loading.value = true
  setTimeout(() => loading.value = false, 2000)
}
function getPreviewText() {
  return `${t('reports.generate.template')}: ${form.value.template}`
}

// provide required emits
// defineEmits(['update:modelValue'])

</script>
<template>
  <UModal :model-value="modelValue" @update:model-value="$emit('update:modelValue', $event)" :title="t('reports.generate.title')" :description="t('reports.generate.subtitle')" size="lg" :close-on-backdrop="true">
    <div class="space-y-5">
      <!-- Report Template -->
      <div class="relative">
        <label class="u-label" for="template">{{ t('reports.generate.template') }}</label>
        <select id="template" v-model="form.template" class="u-input metallic-select" :disabled="loading">
          <option value="executive">{{ t('reports.templates.executive') }}</option>
          <option value="technical">{{ t('reports.templates.technical') }}</option>
          <option value="compliance">{{ t('reports.templates.compliance') }}</option>
          <option value="maintenance">{{ t('reports.templates.maintenance') }}</option>
        </select>
        <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" />
      </div>
      <!-- Date Range -->
      <div class="relative">
        <label class="u-label" for="date-range">{{ t('reports.generate.period') }}</label>
        <select id="date-range" v-model="form.range" class="u-input metallic-select" :disabled="loading">
          <option value="last_24h">{{ t('reports.periods.last_24h') }}</option>
          <option value="last_7d">{{ t('reports.periods.last_7d') }}</option>
          <option value="last_30d">{{ t('reports.periods.last_30d') }}</option>
          <option value="last_90d">{{ t('reports.periods.last_90d') }}</option>
        </select>
        <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" />
      </div>
      <!-- Report Language -->
      <div class="relative">
        <label class="u-label" for="locale">{{ t('reports.generate.language') }}</label>
        <select id="locale" v-model="form.locale" class="u-input metallic-select" :disabled="loading">
          <option value="en-US">{{ t('reports.locales.en') }}</option>
          <option value="ru-RU">{{ t('reports.locales.ru') }}</option>
          <option value="de-DE">{{ t('reports.locales.de') }}</option>
        </select>
        <Icon name="heroicons:chevron-down" class="absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-steel-light pointer-events-none" />
      </div>
      <!-- Custom Title -->
      <div>
        <label class="u-label" for="report-title">{{ t('reports.generate.customTitle') }} <span class="text-text-secondary font-normal">({{ t('ui.optional') }})</span></label>
        <input id="report-title" v-model.trim="form.title" type="text" class="u-input" :placeholder="t('reports.generate.customTitlePlaceholder')" :disabled="loading" maxlength="255" />
      </div>
      <!-- Generation Preview -->
      <div class="rounded-lg bg-success-500/5 border border-success-500/30 p-4">
        <div class="flex items-start gap-3">
          <Icon name="heroicons:document-text" class="h-5 w-5 text-success-500 mt-0.5 shrink-0" />
          <div>
            <p class="text-sm font-medium text-success-900">{{ t('reports.generate.preview') }}</p>
            <p class="text-sm text-success-700 mt-1">{{ getPreviewText() }}</p>
            <div class="flex items-center gap-4 mt-3 text-xs text-success-600">
              <span class="flex items-center gap-1"><Icon name="heroicons:clock" class="h-3 w-3" />~2-5 {{ t('ui.minutes') }}</span>
              <span class="flex items-center gap-1"><Icon name="heroicons:document-arrow-down" class="h-3 w-3" />PDF</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <template #footer>
      <button class="u-btn u-btn-secondary" @click="handleCancel" :disabled="loading" type="button">{{ t('ui.cancel') }}</button>
      <button class="u-btn u-btn-success min-w-[120px]" @click="handleSubmit" :disabled="loading" type="button">
        <Icon v-if="loading" name="heroicons:arrow-path" class="h-4 w-4 animate-spin mr-2" />
        <Icon v-else name="heroicons:document-plus" class="h-4 w-4 mr-2" />
        {{ loading ? t('reports.generate.generating') : t('reports.generate.generateBtn') }}
      </button>
    </template>
  </UModal>
</template>
<style scoped>.metallic-select { background-color: #191d23 !important; color: #edf2fa !important; border-color: #4c596f !important; }</style>
