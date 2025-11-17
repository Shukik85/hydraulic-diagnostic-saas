<template>
  <div class="h-[calc(100vh-8rem)] flex gap-6">
    <!-- ... template unchanged for brevity ... -->
  </div>
</template>

<script setup lang="ts">
import type { ChatMessage, ChatSession } from '~/types/chat'

definePageMeta({
  middleware: ['auth']
})

const { t } = useI18n()

const activeSession = ref<ChatSession | null>(null)
const newMessage = ref('')
const isLoading = ref(false)

const exampleQuestions = [
  {
    icon: 'heroicons:document-magnifying-glass',
    text: t('chat.examples.diagnostics')
  },
  {
    icon: 'heroicons:arrow-trending-up',
    text: t('chat.examples.pressure')
  },
  {
    icon: 'heroicons:wrench-screwdriver',
    text: t('chat.examples.maintenance')
  },
  {
    icon: 'heroicons:fire',
    text: t('chat.examples.temperature')
  }
]

const chatSessions = ref<ChatSession[]>([
  {
    id: 1,
    title: 'Диагностика насосной станции',
    description: 'Проблемы с давлением в HYD-001',
    lastMessage: 'Проверьте фильтр и уплотнения...',
    timestamp: '2 часа назад',
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'Привет! У меня проблемы с насосной станцией HYD-001. Давление нестабильное и падает.',
        timestamp: '14:32'
      },
      {
        id: 2,
        role: 'assistant',
        content: 'Понял. Нестабильность давления в насосной станции может быть вызвана несколькими причинами. Проверьте:\n\n1. Состояние фильтра — загрязнённый фильтр может ограничивать поток\n2. Уплотнения и соединения\n3. Состояние насоса\n\nКакая температура рабочей жидкости?',
        timestamp: '14:33',
        sources: [
          { title: 'Руководство по диагностике', url: '#' },
          { title: 'Техническая документация HYD-001', url: '#' }
        ]
      }
    ]
  }
])

const selectSession = (session: ChatSession): void => {
  activeSession.value = session
}

const startNewSession = (): void => {
  const newSession: ChatSession = {
    id: Date.now(),
    title: 'Новая консультация',
    description: 'Опишите вашу проблему с гидравлической системой',
    lastMessage: '',
    timestamp: 'Новая сессия',
    messages: []
  }
  chatSessions.value.unshift(newSession)
  activeSession.value = newSession
}

const askQuestion = (questionText: string): void => {
  startNewSession()
  newMessage.value = questionText
  sendMessage()
}

const sendMessage = async (): Promise<void> => {
  if (!newMessage.value.trim() || !activeSession.value) return

  isLoading.value = true

  const userMessage: ChatMessage = {
    id: Date.now(),
    role: 'user',
    content: newMessage.value,
    timestamp: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
  }

  activeSession.value.messages.push(userMessage)
  const currentMessage = newMessage.value
  newMessage.value = ''

  setTimeout(() => {
    if (activeSession.value) {
      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Ответ на: "${currentMessage}"\n\nПонял вашу проблему. Рекомендую начать с проверки основных параметров системы.`,
        timestamp: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
      }
      activeSession.value.messages.push(assistantMessage)
      activeSession.value.lastMessage = assistantMessage.content.substring(0, 80) + '...'
      activeSession.value.timestamp = assistantMessage.timestamp
    }
    isLoading.value = false
  }, 1500)
}
</script>
