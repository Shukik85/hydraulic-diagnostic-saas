<script setup lang="ts">
import { computed, ref } from 'vue'

type AppLocale = 'ru' | 'en'

definePageMeta({
  layout: 'blank' as const,
  middleware: []
})

useSeoMeta({
  title: 'AI Hydraulic Diagnostic | SaaS Monitoring Platform',
  description: 'Платформа мониторинга и диагностики гидравлических систем с ИИ и отчётами.',
  ogTitle: 'Hydraulic Diagnostic SaaS',
  ogDescription: 'Платформа для онлайн-мониторинга, аналитики и предупреждения аварий на гидравлических объектах',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

const { locale, setLocale, t } = useI18n()
const showLanguageDropdown = ref(false)

const availableLocales = [
  { code: 'ru' as AppLocale, name: 'Русский' },
  { code: 'en' as AppLocale, name: 'English' }
]

const currentLocale = computed(() =>
  availableLocales.find(l => l.code === (locale.value as AppLocale)) ?? availableLocales[0]
)

const switchLanguage = async (code: string) => {
  await setLocale(code as AppLocale)
  showLanguageDropdown.value = false
}

// Features data
const features = [
  {
    key: 'monitoring',
    icon: 'heroicons:chart-bar',
    iconBg: 'bg-blue-500/20',
    iconColor: 'text-blue-400',
    title: 'landing.features.monitoring.title',
    description: 'landing.features.monitoring.description'
  },
  {
    key: 'analytics',
    icon: 'heroicons:sparkles',
    iconBg: 'bg-green-500/20',
    iconColor: 'text-green-400',
    title: 'landing.features.analytics.title',
    description: 'landing.features.analytics.description'
  },
  {
    key: 'alerts',
    icon: 'heroicons:bell-alert',
    iconBg: 'bg-purple-500/20',
    iconColor: 'text-purple-400',
    title: 'landing.features.alerts.title',
    description: 'landing.features.alerts.description'
  },
  {
    key: 'reports',
    icon: 'heroicons:document-chart-bar',
    iconBg: 'bg-yellow-500/20',
    iconColor: 'text-yellow-400',
    title: 'landing.features.reports.title',
    description: 'landing.features.reports.description'
  },
  {
    key: 'security',
    icon: 'heroicons:lock-closed',
    iconBg: 'bg-red-500/20',
    iconColor: 'text-red-400',
    title: 'landing.features.security.title',
    description: 'landing.features.security.description'
  },
  {
    key: 'integration',
    icon: 'heroicons:cube-transparent',
    iconBg: 'bg-indigo-500/20',
    iconColor: 'text-indigo-400',
    title: 'landing.features.integration.title',
    description: 'landing.features.integration.description'
  }
]
</script>
