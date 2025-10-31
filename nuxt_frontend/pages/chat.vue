<template>
  <div class="h-[calc(100vh-8rem)] flex gap-6">
    <!-- Chat Sessions Sidebar -->
    <div class="w-80 bg-white rounded-lg border border-gray-200 flex flex-col">
      <!-- Header -->
      <div class="p-6 border-b border-gray-200">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-900">{{ t('chat.title') }}</h2>
          <button @click="startNewSession" class="u-btn u-btn-primary u-btn-sm">
            <Icon name="heroicons:plus" class="w-4 h-4" />
          </button>
        </div>
        <p class="text-sm text-gray-600">{{ t('chat.subtitle') }}</p>
      </div>

      <!-- Sessions List -->
      <div class="flex-1 overflow-y-auto">
        <div class="p-4 space-y-2">
          <div v-for="session in chatSessions" :key="session.id" @click="selectSession(session)"
            class="p-4 rounded-lg cursor-pointer u-transition-fast" :class="activeSession?.id === session.id
              ? 'bg-blue-50 border-blue-200 border'
              : 'hover:bg-gray-50 border border-transparent'">
            <h3 class="font-medium text-gray-900 truncate">{{ session.title }}</h3>
            <p class="text-sm text-gray-600 truncate mt-1">{{ session.description }}</p>
            <p class="text-xs text-gray-400 mt-2">{{ session.lastMessage }}</p>
            <span class="text-xs text-gray-400">{{ session.timestamp }}</span>
          </div>
        </div>
      </div>

      <!-- New Session Button (Mobile Alternative) -->
      <div class="p-4 border-t border-gray-200">
        <button @click="startNewSession" class="w-full u-btn u-btn-secondary u-btn-sm">
          <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
          {{ t('chat.newSession.button') }}
        </button>
      </div>
    </div>

    <!-- Chat Area -->
    <div class="flex-1 bg-white rounded-lg border border-gray-200 flex flex-col">
      <template v-if="activeSession">
        <!-- Chat Header -->
        <div class="p-6 border-b border-gray-200">
          <div class="flex items-center gap-3">
            <div
              class="w-10 h-10 bg-linear-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <Icon name="heroicons:chat-bubble-left-ellipsis" class="w-5 h-5 text-white" />
            </div>
            <div class="flex-1 min-w-0">
              <h1 class="u-h4 truncate">{{ activeSession.title }}</h1>
              <p class="text-sm text-gray-600 truncate">
                {{ activeSession.description }}
              </p>
            </div>
            <button class="u-btn u-btn-ghost u-btn-sm" @click="startNewSession">
              <Icon name="heroicons:plus" class="w-4 h-4" />
            </button>
          </div>
        </div>

        <!-- Messages -->
        <div class="flex-1 overflow-y-auto p-6 space-y-4">
          <div v-for="message in activeSession.messages" :key="message.id" class="flex items-start gap-4"
            :class="message.role === 'user' ? 'flex-row-reverse' : 'flex-row'">
            <!-- Avatar -->
            <div class="w-8 h-8 rounded-full flex items-center justify-center shrink-0" :class="message.role === 'user'
              ? 'bg-linear-to-br from-blue-500 to-purple-500'
              : 'bg-linear-to-br from-gray-400 to-gray-500'">
              <Icon :name="message.role === 'user' ? 'heroicons:user' : 'heroicons:cpu-chip'"
                class="w-4 h-4 text-white" />
            </div>

            <!-- Message Content -->
            <div class="flex-1 max-w-2xl">
              <div class="rounded-2xl px-4 py-3" :class="message.role === 'user'
                ? 'bg-linear-to-r from-blue-600 to-purple-600 text-white ml-12'
                : 'bg-gray-50 text-gray-900 mr-12'">
                <p class="whitespace-pre-wrap">{{ message.content }}</p>
              </div>

              <!-- Sources -->
              <div v-if="message.sources && message.sources.length > 0" class="mt-2 space-y-1">
                <p class="text-xs text-gray-500 mb-2">Источники:</p>
                <div v-for="source in message.sources" :key="source.url" class="text-xs">
                  <a :href="source.url" target="_blank" class="text-blue-600 hover:text-blue-500 hover:underline">
                    {{ source.title }}
                  </a>
                </div>
              </div>

              <span class="text-xs text-gray-400 mt-2 block">{{ message.timestamp }}</span>
            </div>
          </div>
        </div>

        <!-- Input Area -->
        <div class="p-6 border-t border-gray-200">
          <form @submit.prevent="sendMessage" class="flex gap-3">
            <div class="flex-1 relative">
              <input v-model="newMessage" type="text" :placeholder="t('chat.placeholder')"
                class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                :disabled="isLoading" />
              <button v-if="newMessage.trim()" type="submit"
                class="absolute right-3 top-1/2 -translate-y-1/2 text-blue-600 hover:text-blue-500 disabled:opacity-50"
                :disabled="isLoading">
                <Icon name="heroicons:paper-airplane" class="w-5 h-5" />
              </button>
            </div>
          </form>

          <!-- Loading state -->
          <div v-if="isLoading" class="flex items-center gap-2 text-sm text-gray-600 mt-2">
            <div class="w-4 h-4 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin"></div>
            {{ t('chat.thinking') }}
          </div>
        </div>
      </template>

      <!-- Empty State -->
      <template v-else>
        <div class="flex-1 flex items-center justify-center p-8">
          <div class="text-center max-w-md">
            <div
              class="w-16 h-16 bg-linear-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <Icon name="heroicons:chat-bubble-left-ellipsis" class="w-8 h-8 text-white" />
            </div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">{{ t('chat.newSession.title') }}</h3>
            <p class="text-gray-600 mb-6">Начните новую консультацию и задайте вопрос о ваших гидравлических системах.
            </p>
            <button @click="startNewSession" class="u-btn u-btn-primary">
              <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
              {{ t('chat.newSession.button') }}
            </button>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ChatMessage, ChatSession } from '~/types/api'

// Page metadata
definePageMeta({
  middleware: ['auth']
})

// Composables
const { t } = useI18n()

// State with proper typing
const activeSession = ref(null)
const newMessage = ref('')
const isLoading = ref(false)

// Mock chat sessions data with proper typing
const chatSessions = ref([
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

// Methods
const selectSession = (session: ChatSession) => {
  activeSession.value = session
}

const startNewSession = () => {
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

const sendMessage = async () => {
  if (!newMessage.value.trim() || !activeSession.value) return

  isLoading.value = true

  // Add user message
  const userMessage: ChatMessage = {
    id: Date.now(),
    role: 'user',
    content: newMessage.value,
    timestamp: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
  }

  activeSession.value.messages.push(userMessage)
  const currentMessage = newMessage.value
  newMessage.value = ''

  // Simulate AI response
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

// Initialize with first session if available - safe assignment
const firstSession = chatSessions.value[0] || null
if (firstSession) {
  activeSession.value = firstSession
}
</script>