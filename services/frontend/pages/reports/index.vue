<script setup lang="ts">
import { ref, computed } from 'vue'

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

const showGenerateModal = ref(false)
const isGenerating = ref(false)
const loading = ref(false)
const form = ref({ template: 'executive', period: 'last_7d', locale: 'ru', customTitle: '' })

const reports = ref([
  { id: 1, title: 'Executive Summary - Weekly Report', description: 'Краткий обзор состояния гидравлических систем', createdAt: '2 часа назад', period: 'Последние 7 дней', severity: 'low', status: 'completed' },
  { id: 2, title: 'Technical Analysis - System HYD-001', description: 'Детальный технический анализ насосной станции', createdAt: '1 день назад', period: 'Последние 30 дней', severity: 'medium', status: 'completed' }
])

const reportTemplates = [
  { key: 'executive', name: t('reports.templates.execShort'), description: t('reports.templates.executive') },
  { key: 'technical', name: t('reports.templates.techShort'), description: t('reports.templates.technical') },
  { key: 'compliance', name: t('reports.templates.compShort'), description: t('reports.templates.compliance') },
  { key: 'maintenance', name: t('reports.templates.maintShort'), description: t('reports.templates.maintenance') }
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

const getStatusVariant = (s: string): 'success' | 'default' | 'warning' | 'destructive' => {
  const variants: Record<string, 'success' | 'default' | 'warning' | 'destructive'> = {
    completed: 'success',
    in_progress: 'default',
    pending: 'warning',
    failed: 'destructive'
  }
  return variants[s] || 'default'
}

const generateReport = async (): Promise<void> => {
  isGenerating.value = true
  setTimeout(() => {
    const templateName = reportTemplates.find(t => t.key === form.value.template)?.name || ''
    const newReport = {
      id: Date.now(),
      title: form.value.customTitle || `${templateName} - ${new Date().toLocaleDateString('ru-RU')}`,
      description: `Отчёт за ...`,
      createdAt: 'Только что',
      period: t(`reports.periods.${form.value.period}`),
      severity: 'low',
      status: 'completed'
    }
    reports.value.unshift(newReport)
    showGenerateModal.value = false
    isGenerating.value = false
  }, 2000)
}

const viewReport = (reportId: number): void => {
  navigateTo(`/reports/${reportId}`)
}

const downloadReport = (reportId: number): void => {
  console.log('Downloading report:', reportId)
}
</script>
