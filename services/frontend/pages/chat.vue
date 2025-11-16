<template>
  <div class="h-[calc(100vh-8rem)] flex gap-6">
    <!-- Chat Sessions Sidebar -->
    <div class="w-80 card-glass flex flex-col">
      <!-- Header -->
      <div class="p-6 border-b border-steel-700/50">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-white">{{ t('chat.title') }}</h2>
          <UButton 
            size="icon"
            @click="startNewSession"
          >
            <Icon name="heroicons:plus" class="w-5 h-5" />
          </UButton>
        </div>
        <p class="text-sm text-steel-shine">{{ t('chat.subtitle') }}</p>
      </div>

      <!-- Sessions List -->
      <div class="flex-1 overflow-y-auto scrollbar-thin">
        <div class="p-4 space-y-2">
          <div 
            v-for="session in chatSessions" 
            :key="session.id" 
            class="p-4 rounded-lg cursor-pointer transition-smooth"
            :class="activeSession?.id === session.id
              ? 'bg-primary-600/20 border border-primary-500/50'
              : 'hover:bg-steel-800/30 border border-transparent'"
            @click="selectSession(session)"
          >
            <h3 class="font-medium text-white truncate">{{ session.title }}</h3>
            <p class="text-sm text-steel-shine truncate mt-1">{{ session.description }}</p>
            <div class="flex items-center justify-between mt-2">
              <span class="text-xs text-steel-400">{{ session.timestamp }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- New Session Button (Alternative) -->
      <div class="p-4 border-t border-steel-700/50">
        <UButton 
          variant="secondary"
          class="w-full"
          @click="startNewSession"
        >
          <Icon name="heroicons:plus" class="w-5 h-5 mr-2" />
          {{ t('chat.newSession.button') }}
        </UButton>
      </div>
    </div>

    <!-- Chat Area -->
    <div class="flex-1 card-glass flex flex-col">
      <template v-if="activeSession && activeSession.messages.length > 0">
        <!-- Chat Header -->
        <div class="p-6 border-b border-steel-700/50">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-500 rounded-full flex items-center justify-center">
              <Icon name="heroicons:chat-bubble-left-ellipsis" class="w-5 h-5 text-white" />
            </div>
            <div class="flex-1 min-w-0">
              <h1 class="text-lg font-bold text-white truncate">{{ activeSession.title }}</h1>
              <p class="text-sm text-steel-shine truncate">{{ activeSession.description }}</p>
            </div>
            <UButton 
              variant="ghost" 
              size="icon"
              @click="startNewSession"
            >
              <Icon name="heroicons:plus" class="w-5 h-5" />
            </UButton>
          </div>
        </div>

        <!-- Messages -->
        <div class="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin">
          <div 
            v-for="message in activeSession.messages" 
            :key="message.id" 
            class="flex items-start gap-4"
            :class="message.role === 'user' ? 'flex-row-reverse' : 'flex-row'"
          >
            <!-- Avatar -->
            <div 
              class="w-8 h-8 rounded-full flex items-center justify-center shrink-0" 
              :class="message.role === 'user'
                ? 'bg-gradient-to-br from-primary-500 to-purple-500'
                : 'bg-gradient-to-br from-steel-600 to-steel-700'"
            >
              <Icon 
                :name="message.role === 'user' ? 'heroicons:user' : 'heroicons:cpu-chip'"
                class="w-4 h-4 text-white" 
              />
            </div>

            <!-- Message Content -->
            <div class="flex-1 max-w-2xl">
              <div 
                class="rounded-2xl px-4 py-3" 
                :class="message.role === 'user'
                  ? 'bg-gradient-to-r from-primary-600 to-purple-600 text-white ml-12'
                  : 'bg-steel-800/50 text-white mr-12'"
              >
                <p class="whitespace-pre-wrap">{{ message.content }}</p>
              </div>

              <!-- Sources -->
              <div 
                v-if="message.sources && message.sources.length > 0" 
                class="mt-2 space-y-1"
              >
                <p class="text-xs text-steel-400 mb-2">Источники:</p>
                <div 
                  v-for="source in message.sources" 
                  :key="source.url" 
                  class="text-xs"
                >
                  <a 
                    :href="source.url" 
                    target="_blank" 
                    class="text-primary-400 hover:text-primary-300 hover:underline"
                  >
                    {{ source.title }}
                  </a>
                </div>
              </div>

              <span class="text-xs text-steel-400 mt-2 block">{{ message.timestamp }}</span>
            </div>
          </div>
        </div>

        <!-- Input Area -->
        <div class="p-6 border-t border-steel-700/50">
          <form @submit.prevent="sendMessage" class="flex gap-3">
            <div class="flex-1 relative">
              <input 
                v-model="newMessage" 
                type="text" 
                :placeholder="t('chat.placeholder')"
                class="input-text pr-12"
                :disabled="isLoading" 
              />
              <button 
                v-if="newMessage.trim()" 
                type="submit"
                class="absolute right-3 top-1/2 -translate-y-1/2 text-primary-400 hover:text-primary-300 disabled:opacity-50"
                :disabled="isLoading"
                aria-label="Отправить"
              >
                <Icon name="heroicons:paper-airplane" class="w-5 h-5" />
              </button>
            </div>
          </form>

          <!-- Loading state -->
          <div 
            v-if="isLoading" 
            class="flex items-center gap-2 text-sm text-steel-shine mt-2"
          >
            <div class="w-4 h-4 border-2 border-steel-600 border-t-primary-400 rounded-full animate-spin" />
            {{ t('chat.thinking') }}
          </div>
        </div>
      </template>

      <!-- Welcome Screen with Examples -->
      <template v-else>
        <div class="flex-1 flex items-center justify-center p-8">
          <div class="text-center max-w-2xl">
            <div class="w-20 h-20 bg-gradient-to-br from-primary-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Icon name="heroicons:chat-bubble-left-right" class="w-10 h-10 text-white" />
            </div>
            
            <h2 class="text-3xl font-bold text-white mb-4">
              {{ t('chat.welcome.title') }}
            </h2>
            
            <p class="text-steel-shine mb-8 text-lg">
              {{ t('chat.welcome.description') }}
            </p>

            <!-- Example Questions -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-8">
              <button
                v-for="(example, index) in exampleQuestions"
                :key="index"
                class="p-4 rounded-lg card-glass border border-steel-700/50 hover:border-primary-500/50 hover:bg-steel-800/80 transition-all text-left"
                @click="askQuestion(example.text)"
              >
                <Icon :name="example.icon" class="w-5 h-5 text-primary-400 mb-2" />
                <p class="text-sm text-white">{{ example.text }}</p>
              </button>
            </div>

            <UButton 
              size="lg"
              @click="startNewSession"
            >
              <Icon name="heroicons:plus" class="w-5 h-5 mr-2" />
              {{ t('chat.newSession.button') }}
            </UButton>
          </div>
        </div>
      </template>
    </div>
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

const askQuestion = (questionText: string) => {
  startNewSession()
  newMessage.value = questionText
  sendMessage()
}

const sendMessage = async () => {
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
