<script setup lang="ts">
import { ref, computed } from 'vue'

useSeoMeta({
  title: 'Диагностика | Hydraulic Diagnostic SaaS',
  description: 'Страница диагностики: запуск новых и просмотр истории гидроанализов с AI-выводом. Быстрый доступ к KPI, статусу и рекомендациям.',
  ogTitle: 'Diagnostics | Hydraulic Diagnostic SaaS',
  ogDescription: 'Diagnostic history, KPI, recommendations and AI insights for hydraulic systems',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

interface ActiveSession {
  id: number
  name: string
  equipment: string
  progress: number
  startedAt: string
}

interface DiagnosticUIResult {
  id: number
  name: string
  equipment: string
  score: number
  issuesFound: number
  completedAt: string
  status: 'completed' | 'warning' | 'error' | 'processing'
  duration: string
}

definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

const showRunModal = ref(false)
const showResultsModal = ref(false)
const isRunning = ref(false)
const loading = ref(false)
const selectedResult = ref<DiagnosticUIResult | null>(null)

const activeSessions = ref<ActiveSession[]>([])
const recentResults = ref<DiagnosticUIResult[]>([
  {
    id: 1,
    name: 'Full System Analysis - HYD-001',
    equipment: 'HYD-001 - Pump Station A',
    score: 92,
    issuesFound: 1,
    completedAt: '2 часа назад',
    status: 'completed',
    duration: '4.2 min'
  },
  {
    id: 2,
    name: 'Pressure Check - HYD-002',
    equipment: 'HYD-002 - Hydraulic Motor B',
    score: 78,
    issuesFound: 3,
    completedAt: '6 часов назад',
    status: 'warning',
    duration: '2.8 min'
  }
])

const startDiagnostic = async (data: any): Promise<void> => {
  isRunning.value = true

  const session: ActiveSession = {
    id: Date.now(),
    name: `New Diagnostic - ${data.equipment}`,
    equipment: data.equipment,
    progress: 0,
    startedAt: 'just now'
  }

  activeSessions.value.push(session)
  showRunModal.value = false

  const interval = setInterval(() => {
    session.progress += Math.random() * 20
    if (session.progress >= 100) {
      session.progress = 100
      clearInterval(interval)

      setTimeout(() => {
        activeSessions.value = activeSessions.value.filter(s => s.id !== session.id)
        isRunning.value = false

        const newResult: DiagnosticUIResult = {
          id: session.id,
          name: session.name,
          equipment: session.equipment,
          score: Math.floor(Math.random() * 30) + 70,
          issuesFound: Math.floor(Math.random() * 4),
          completedAt: 'just now',
          status: 'completed',
          duration: `${Math.floor(Math.random() * 3) + 2}.${Math.floor(Math.random() * 9)} min`
        }
        recentResults.value.unshift(newResult)
      }, 1000)
    }
  }, 500)
}

const cancelSession = (sessionId: number): void => {
  activeSessions.value = activeSessions.value.filter(s => s.id !== sessionId)
  if (activeSessions.value.length === 0) {
    isRunning.value = false
  }
}

const viewResult = (resultId: number): void => {
  const result = recentResults.value.find(r => r.id === resultId)
  if (result) {
    selectedResult.value = result
    showResultsModal.value = true
  }
}

const getStatusVariant = (status: string): 'default' | 'success' | 'warning' | 'destructive' => {
  const variants: Record<string, 'default' | 'success' | 'warning' | 'destructive'> = {
    completed: 'success',
    warning: 'warning',
    error: 'destructive',
    processing: 'default'
  }
  return variants[status] || 'default'
}

const getStatusIcon = (status: string): string => {
  const icons: Record<string, string> = {
    completed: 'heroicons:check-circle',
    warning: 'heroicons:exclamation-triangle',
    error: 'heroicons:x-circle',
    processing: 'heroicons:arrow-path'
  }
  return icons[status] || 'heroicons:question-mark-circle'
}
</script>
