<script setup lang="ts">
import type { ChatMessage, ChatSession } from '~/types/chat'
import { ref } from '#imports'

definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

useSeoMeta({
  title: 'Чат консультаций | Hydraulic Diagnostic SaaS',
  description: 'Интерактивный чат с поддержкой ИИ: быстрые консультации по гидравлическим системам, примеры вопросов, поиск по базе знаний.',
  ogTitle: 'Chat | Hydraulic Diagnostic SaaS',
  ogDescription: 'Interactive consultation chat for hydraulic anomaly troubleshooting & expert QA',
  ogType: 'website',
  twitterCard: 'summary_large_image'
})

const activeSession = ref<ChatSession | null>(null)
const newMessage = ref('')
const isLoading = ref(false)
const chatSessions = ref<ChatSession[]>([])

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

<template>
  <div class="chat-container">
    <h1 class="u-h2 mb-6">{{ t('chat.title', 'AI Консультации') }}</h1>
    <!-- Остальная часть template остается без изменений -->
  </div>
</template>
