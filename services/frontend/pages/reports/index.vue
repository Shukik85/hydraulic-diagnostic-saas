<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('reports.title') }}</h1>
        <p class="text-steel-shine mt-2">{{ t('reports.subtitle') }}</p>
      </div>
      <UButton 
        size="lg"
        @click="showGenerateModal = true"
      >
        <Icon name="heroicons:document-plus" class="w-5 h-5 mr-2" />
        {{ t('reports.generate.generateBtn') }}
      </UButton>
    </div>

    <!-- Zero State -->
    <UZeroState
      v-if="!loading && reports.length === 0"
      icon-name="heroicons:document-text"
      :title="t('reports.empty.title')"
      :description="t('reports.empty.description')"
      action-icon="heroicons:document-plus"
      :action-text="t('reports.empty.action')"
      @action="showGenerateModal = true"
    />

    <!-- Reports List -->
    <div v-else class="space-y-4">
      <div 
        v-for="report in reports" 
        :key="report.id" 
        class="card-interactive p-6"
        role="button"
        tabindex="0"
        @click="viewReport(report.id)"
        @keydown.enter="viewReport(report.id)"
      >
        <div class="flex items-start gap-4">
          <!-- Icon -->
          <div class="w-12 h-12 rounded-lg bg-primary-600/10 flex items-center justify-center flex-shrink-0">
            <Icon name="heroicons:document-text" class="w-6 h-6 text-primary-400" />
          </div>

          <!-- Content -->
          <div class="flex-1 min-w-0">
            <div class="flex items-start justify-between gap-4 mb-2">
              <h3 class="font-semibold text-white text-lg truncate">{{ report.title }}</h3>
              <UBadge :variant="getStatusVariant(report.status)">
                <Icon :name="getStatusIcon(report.status)" class="w-3 h-3" />
                {{ getStatusText(report.status) }}
              </UBadge>
            </div>

            <p class="text-steel-shine mb-3">{{ report.description }}</p>

            <!-- Meta Info -->
            <div class="flex flex-wrap items-center gap-4 text-xs text-steel-400">
              <div class="flex items-center gap-1.5">
                <Icon name="heroicons:clock" class="w-3 h-3" />
                <span>Генерация: {{ report.createdAt }}</span>
              </div>
              <div class="flex items-center gap-1.5">
                <Icon name="heroicons:calendar" class="w-3 h-3" />
                <span>Период: {{ report.period }}</span>
              </div>
              <div class="flex items-center gap-1.5">
                <div 
                  class="w-2 h-2 rounded-full"
                  :class="getSeverityColor(report.severity)"
                />
                <span>{{ getSeverityText(report.severity) }}</span>
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex items-center gap-2 flex-shrink-0">
            <UButton 
              variant="ghost" 
              size="icon"
              @click.stop="viewReport(report.id)"
              aria-label="Просмотр"
            >
              <Icon name="heroicons:eye" class="w-5 h-5" />
            </UButton>
            <UButton 
              variant="ghost" 
              size="icon"
              @click.stop="downloadReport(report.id)"
              aria-label="Скачать"
            >
              <Icon name="heroicons:arrow-down-tray" class="w-5 h-5" />
            </UButton>
          </div>
        </div>
      </div>
    </div>

    <!-- Generate Report Modal -->
    <UDialog v-model="showGenerateModal">
      <UDialogContent class="max-w-2xl">
        <UDialogHeader>
          <UDialogTitle>{{ t('reports.generate.title') }}</UDialogTitle>
          <UDialogDescription>{{ t('reports.generate.subtitle') }}</UDialogDescription>
        </UDialogHeader>

        <form @submit.prevent="generateReport" class="space-y-6">
          <!-- Template Selection -->
          <UFormGroup
            :label="t('reports.generate.template')"
            helper="Выберите тип отчёта в зависимости от аудитории"
            required
          >
            <div class="grid gap-3 sm:grid-cols-2">
              <label 
                v-for="template in reportTemplates" 
                :key="template.key" 
                class="relative"
              >
                <input 
                  v-model="form.template" 
                  :value="template.key" 
                  type="radio" 
                  class="sr-only peer" 
                />
                <div class="p-4 card-glass border border-steel-700/50 rounded-lg cursor-pointer transition-all peer-checked:border-primary-500 peer-checked:bg-primary-600/10 hover:border-steel-600">
                  <div class="font-medium text-white">{{ template.name }}</div>
                  <div class="text-sm text-steel-shine mt-1">{{ template.description }}</div>
                </div>
              </label>
            </div>
          </UFormGroup>

          <!-- Period -->
          <UFormGroup
            :label="t('reports.generate.period')"
            helper="Временной диапазон для анализа данных"
            required
          >
            <USelect v-model="form.period">
              <option value="last_24h">{{ t('reports.periods.last_24h') }}</option>
              <option value="last_7d">{{ t('reports.periods.last_7d') }}</option>
              <option value="last_30d">{{ t('reports.periods.last_30d') }}</option>
              <option value="last_90d">{{ t('reports.periods.last_90d') }}</option>
            </USelect>
          </UFormGroup>

          <!-- Language -->
          <UFormGroup
            :label="t('reports.generate.language')"
            helper="Язык генерируемого отчёта"
          >
            <USelect v-model="form.locale">
              <option value="ru">{{ t('reports.locales.ru') }}</option>
              <option value="en">{{ t('reports.locales.en') }}</option>
              <option value="de">{{ t('reports.locales.de') }}</option>
            </USelect>
          </UFormGroup>

          <!-- Custom Title -->
          <UFormGroup
            :label="t('reports.generate.customTitle')"
            helper="Оставьте пустым для автоматического названия"
          >
            <UInput 
              v-model="form.customTitle" 
              :placeholder="t('reports.generate.customTitlePlaceholder')" 
            />
          </UFormGroup>
        </form>

        <UDialogFooter>
          <UButton 
            variant="secondary"
            @click="showGenerateModal = false"
          >
            {{ t('ui.cancel') }}
          </UButton>
          <UButton 
            :disabled="isGenerating"
            @click="generateReport"
          >
            <Icon 
              v-if="isGenerating" 
              name="heroicons:arrow-path" 
              class="w-5 h-5 mr-2 animate-spin" 
            />
            <Icon 
              v-else 
              name="heroicons:document-plus" 
              class="w-5 h-5 mr-2" 
            />
            {{ isGenerating ? t('reports.generate.generating') : t('reports.generate.generateBtn') }}
          </UButton>
        </UDialogFooter>
      </UDialogContent>
    </UDialog>
  </div>
</template>

<script setup lang="ts">
definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

const showGenerateModal = ref(false)
const isGenerating = ref(false)
const loading = ref(false)

const form = ref({ 
  template: 'executive', 
  period: 'last_7d', 
  locale: 'ru', 
  customTitle: '' 
})

const reports = ref([
  { 
    id: 1, 
    title: 'Executive Summary - Weekly Report', 
    description: 'Краткий обзор состояния гидравлических систем', 
    createdAt: '2 часа назад', 
    period: 'Последние 7 дней', 
    severity: 'low' as const, 
    status: 'completed' as const 
  },
  { 
    id: 2, 
    title: 'Technical Analysis - System HYD-001', 
    description: 'Детальный технический анализ насосной станции', 
    createdAt: '1 день назад', 
    period: 'Последние 30 дней', 
    severity: 'medium' as const, 
    status: 'completed' as const 
  }
])

const reportTemplates = [
  { 
    key: 'executive', 
    name: t('reports.templates.execShort'), 
    description: t('reports.templates.executive') 
  },
  { 
    key: 'technical', 
    name: t('reports.templates.techShort'), 
    description: t('reports.templates.technical') 
  },
  { 
    key: 'compliance', 
    name: t('reports.templates.compShort'), 
    description: t('reports.templates.compliance') 
  },
  { 
    key: 'maintenance', 
    name: t('reports.templates.maintShort'), 
    description: t('reports.templates.maintenance') 
  }
]

const getSeverityColor = (s: string): string => {
  const colors: Record<string, string> = {
    low: 'bg-success-500',
    medium: 'bg-yellow-500',
    high: 'bg-orange-500',
    critical: 'bg-red-500'
  }
  return colors[s] || 'bg-steel-500'
}

const getSeverityText = (s: string): string => {
  const texts: Record<string, string> = {
    low: t('reports.severity.low'),
    medium: t('reports.severity.medium'),
    high: t('reports.severity.high'),
    critical: t('reports.severity.critical')
  }
  return texts[s] || s
}

const getStatusVariant = (s: string): 'success' | 'default' | 'warning' | 'destructive' => {
  const variants: Record<string, 'success' | 'default' | 'warning' | 'destructive'> = {
    completed: 'success',
    in_progress: 'default',
    pending: 'warning',
    failed: 'destructive'
  }
  return variants[s] || 'default'
}

const getStatusIcon = (s: string): string => {
  const icons: Record<string, string> = {
    completed: 'heroicons:check-circle',
    in_progress: 'heroicons:arrow-path',
    pending: 'heroicons:clock',
    failed: 'heroicons:x-circle'
  }
  return icons[s] || 'heroicons:question-mark-circle'
}

const getStatusText = (s: string): string => {
  const texts: Record<string, string> = {
    completed: t('reports.status.completed'),
    in_progress: t('reports.status.in_progress'),
    pending: t('reports.status.pending'),
    failed: t('reports.status.failed')
  }
  return texts[s] || s
}

const generateReport = async () => {
  isGenerating.value = true
  setTimeout(() => {
    const templateName = reportTemplates.find(t => t.key === form.value.template)?.name || ''
    const periodText = t(`reports.periods.${form.value.period}`)
    
    const newReport = { 
      id: Date.now(), 
      title: form.value.customTitle || `${templateName} - ${new Date().toLocaleDateString('ru-RU')}`, 
      description: `Отчёт за ${periodText}`, 
      createdAt: 'Только что', 
      period: periodText, 
      severity: 'low' as const, 
      status: 'completed' as const 
    }
    
    reports.value.unshift(newReport)
    showGenerateModal.value = false
    isGenerating.value = false
  }, 2000)
}

const viewReport = (reportId: number) => {
  navigateTo(`/reports/${reportId}`)
}

const downloadReport = (reportId: number) => {
  console.log('Downloading report:', reportId)
  // TODO: Implement download logic
}
</script>
