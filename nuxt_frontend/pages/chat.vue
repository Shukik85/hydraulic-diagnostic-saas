<template>
  <div class="flex h-[calc(100vh-8rem)] bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden">
    <!-- Sidebar: Chat Sessions -->
    <div class="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      <!-- Sidebar Header -->
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="u-flex-between mb-4">
          <h2 class="u-h4">AI Assistant</h2>
          <button @click="showNewSessionModal = true" class="u-btn u-btn-primary u-btn-sm">
            <Icon name="heroicons:plus" class="w-4 h-4 mr-1" />
            New
          </button>
        </div>
        <p class="u-body-sm text-gray-600 dark:text-gray-300">
          Intelligent diagnostic support and analysis
        </p>
      </div>

      <!-- Sessions List -->
      <div class="flex-1 overflow-y-auto p-4 space-y-3">
        <div
          v-for="session in chatSessions"
          :key="session.id"
          @click="selectSession(session)"
          class="p-4 rounded-lg cursor-pointer u-transition-fast"
          :class="activeSession?.id === session.id 
            ? 'bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700' 
            : 'hover:bg-gray-50 dark:hover:bg-gray-700 border border-transparent'"
        >
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full u-flex-center flex-shrink-0">
              <Icon name="heroicons:chat-bubble-left" class="w-4 h-4 text-white" />
            </div>
            <div class="flex-1 min-w-0">
              <h3 class="font-medium text-gray-900 dark:text-white truncate mb-1">
                {{ session.title }}
              </h3>
              <p class="u-body-sm text-gray-500 dark:text-gray-400 line-clamp-2 mb-2">
                {{ session.lastMessage || session.description }}
              </p>
              <div class="u-flex-between">
                <span class="text-xs text-gray-400 dark:text-gray-500">
                  {{ formatTimestamp(session.timestamp) }}
                </span>
                <div class="flex items-center gap-1">
                  <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span class="text-xs text-gray-400 dark:text-gray-500">{{ session.messages.length }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col bg-white dark:bg-gray-800">
      <!-- Chat Header -->
      <div v-if="activeSession" class="p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="u-flex-between">
          <div>
            <h1 class="u-h4">{{ activeSession.title }}</h1>
            <p class="u-body-sm text-gray-500 dark:text-gray-400 mt-1">
              {{ activeSession.description }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <button class="u-btn u-btn-ghost u-btn-sm">
              <Icon name="heroicons:ellipsis-horizontal" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <!-- Messages Area -->
      <div class="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900">
        <div v-if="activeSession" class="max-w-4xl mx-auto p-6 space-y-6">
          <div
            v-for="message in activeSession.messages"
            :key="message.id"
            class="flex items-start gap-4"
            :class="message.role === 'user' ? 'flex-row-reverse' : 'flex-row'"
          >
            <!-- Avatar -->
            <div class="flex-shrink-0">
              <div 
                v-if="message.role === 'assistant'"
                class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full u-flex-center"
              >
                <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
              </div>
              <div 
                v-else
                class="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-700 rounded-full u-flex-center"
              >
                <Icon name="heroicons:user" class="w-5 h-5 text-white" />
              </div>
            </div>

            <!-- Message Bubble -->
            <div 
              class="max-w-2xl"
              :class="message.role === 'user' 
                ? 'bg-blue-600 text-white rounded-l-2xl rounded-tr-2xl p-4' 
                : 'u-card p-4'"
            >
              <div class="whitespace-pre-wrap u-body" 
                   :class="message.role === 'user' ? 'text-white' : 'text-gray-900 dark:text-white'">
                {{ message.content }}
              </div>

              <!-- Sources -->
              <div v-if="message.sources?.length" class="mt-3 pt-3 border-t" 
                   :class="message.role === 'user' ? 'border-blue-500' : 'border-gray-200 dark:border-gray-600'">
                <p class="text-xs mb-2" 
                   :class="message.role === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'">
                  Sources:
                </p>
                <div class="space-y-1">
                  <a
                    v-for="source in message.sources"
                    :key="source.url"
                    :href="source.url"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="block text-xs hover:underline"
                    :class="message.role === 'user' ? 'text-blue-100' : 'text-blue-600 dark:text-blue-400'"
                  >
                    <Icon name="heroicons:document-text" class="w-3 h-3 mr-1 inline" />
                    {{ source.title }}
                  </a>
                </div>
              </div>

              <p class="text-xs mt-3 opacity-70" 
                 :class="message.role === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'">
                {{ formatTimestamp(message.timestamp) }}
              </p>
            </div>
          </div>

          <!-- Loading indicator -->
          <div v-if="isLoading" class="flex items-start gap-4">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full u-flex-center">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div class="u-card p-4">
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <span class="u-body-sm text-gray-500 dark:text-gray-400 ml-2">AI is thinking...</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-else class="flex-1 u-flex-center">
          <div class="text-center max-w-md">
            <Icon name="heroicons:chat-bubble-left-ellipsis" class="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <h3 class="u-h5 text-gray-500 dark:text-gray-400 mb-2">
              Select a chat or create a new one
            </h3>
            <p class="u-body text-gray-400 dark:text-gray-500 mb-6">
              AI assistant will help with diagnostics and system optimization
            </p>
            <button @click="showNewSessionModal = true" class="u-btn u-btn-primary u-btn-md">
              <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
              Start New Chat
            </button>
          </div>
        </div>
      </div>

      <!-- Message Input -->
      <div v-if="activeSession" class="border-t border-gray-200 dark:border-gray-700 p-6">
        <form @submit.prevent="sendMessage" class="flex items-end gap-4">
          <div class="flex-1">
            <textarea
              v-model="newMessage"
              :disabled="isLoading"
              placeholder="Ask about diagnostics, maintenance, or system optimization..."
              rows="3"
              class="u-input resize-none"
              @keydown.meta.enter.prevent="sendMessage"
              @keydown.ctrl.enter.prevent="sendMessage"
            ></textarea>
          </div>
          <button
            type="submit"
            :disabled="!newMessage.trim() || isLoading"
            class="u-btn u-btn-primary u-btn-md"
          >
            <Icon v-if="!isLoading" name="heroicons:paper-airplane" class="w-4 h-4 mr-2" />
            <div v-else class="u-spinner w-4 h-4 mr-2"></div>
            {{ isLoading ? 'Sending...' : 'Send' }}
          </button>
        </form>
        <p class="u-body-sm text-gray-500 dark:text-gray-400 mt-2">
          Press Cmd+Enter to send • Powered by AI
        </p>
      </div>
    </div>

    <!-- New Session Modal -->
    <div v-if="showNewSessionModal" class="fixed inset-0 bg-black/50 z-50 u-flex-center p-4" @click="showNewSessionModal = false">
      <div class="u-card max-w-md w-full" @click.stop>
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="u-h4">New Consultation</h3>
          <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
            Describe your diagnostic task or question
          </p>
        </div>

        <form @submit.prevent="createNewSession" class="p-6 space-y-4">
          <div>
            <label class="u-label">Session Title</label>
            <input
              v-model="newSessionTitle"
              type="text"
              required
              class="u-input"
              placeholder="e.g. HYD-002 Optimization Analysis"
              autofocus
            />
          </div>

          <div class="flex gap-3">
            <button
              type="button"
              @click="showNewSessionModal = false"
              class="u-btn u-btn-secondary flex-1"
            >
              Cancel
            </button>
            <button
              type="submit"
              :disabled="!newSessionTitle.trim()"
              class="u-btn u-btn-primary flex-1"
            >
              <Icon name="heroicons:plus" class="w-4 h-4 mr-2" />
              Create Session
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  title: 'AI Chat',
  layout: 'dashboard', // FIXED: changed from 'default' to 'dashboard'
  middleware: ['auth']
})

interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: { title: string; url: string }[]
}

interface ChatSession {
  id: number
  title: string
  description: string
  lastMessage: string
  timestamp: string
  messages: ChatMessage[]
}

// State
const activeSession = ref<ChatSession | null>(null)
const newMessage = ref('')
const isLoading = ref(false)
const showNewSessionModal = ref(false)
const newSessionTitle = ref('')

// Demo chat sessions
const chatSessions = ref<ChatSession[]>([
  {
    id: 1,
    title: 'HYD-001 System Analysis',
    description: 'Cooling system diagnostics',
    lastMessage: 'System running stable, but I recommend checking...',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'Analyze the condition of HYD-001 system',
        timestamp: new Date(Date.now() - 3600000).toISOString()
      },
      {
        id: 2,
        role: 'assistant',
        content: 'HYD-001 system is operating stably. Temperature: 45.2°C, pressure: 150.8 bar. I recommend checking filters within 2 weeks.',
        timestamp: new Date(Date.now() - 3580000).toISOString(),
        sources: [
          { title: 'HYD-001 Technical Specifications', url: '/docs/hyd-001-specs.pdf' },
          { title: 'Maintenance History', url: '/maintenance/hyd-001-history' }
        ]
      }
    ]
  },
  {
    id: 2,
    title: 'Energy Optimization',
    description: 'Cost reduction consultation',
    lastMessage: 'I recommend installing variable frequency drives...',
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'How can I reduce energy consumption of hydraulic systems?',
        timestamp: new Date(Date.now() - 7200000).toISOString()
      },
      {
        id: 2,
        role: 'assistant',
        content: 'I recommend installing variable frequency drives with adjustable speed control and energy recovery systems. This can reduce consumption by 25-40%.',
        timestamp: new Date(Date.now() - 7180000).toISOString(),
        sources: [
          { title: 'Energy Efficiency Guide', url: '/docs/energy-guide.pdf' }
        ]
      }
    ]
  }
])

// Methods
const selectSession = (session: ChatSession) => {
  activeSession.value = session
}

const sendMessage = async () => {
  if (!newMessage.value.trim() || !activeSession.value || isLoading.value) return

  const message = newMessage.value.trim()
  newMessage.value = ''
  isLoading.value = true

  // Add user message
  const userMessage: ChatMessage = {
    id: Date.now(),
    role: 'user',
    content: message,
    timestamp: new Date().toISOString()
  }

  activeSession.value.messages.push(userMessage)

  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Add assistant response
    const assistantMessage: ChatMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: `Based on your question: "${message}". I've analyzed your request and can help with hydraulic system diagnostics. Here are my recommendations...`,
      timestamp: new Date().toISOString(),
      sources: [
        { title: 'Technical Documentation', url: '/docs/hydraulic-systems.pdf' },
        { title: 'Best Practices Guide', url: '/docs/best-practices.pdf' }
      ]
    }

    if (activeSession.value) {
      activeSession.value.messages.push(assistantMessage)
      activeSession.value.lastMessage = assistantMessage.content.substring(0, 80) + '...'
      activeSession.value.timestamp = assistantMessage.timestamp
    }
  } catch (error) {
    console.error('Chat error:', error)
  } finally {
    isLoading.value = false
  }
}

const createNewSession = () => {
  if (!newSessionTitle.value.trim()) return

  const newSession: ChatSession = {
    id: Date.now(),
    title: newSessionTitle.value.trim(),
    description: 'New diagnostic consultation',
    lastMessage: '',
    timestamp: new Date().toISOString(),
    messages: []
  }

  chatSessions.value.unshift(newSession)
  activeSession.value = newSession
  newSessionTitle.value = ''
  showNewSessionModal.value = false
}

const formatTimestamp = (timestamp: string) => {
  try {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  } catch {
    return 'Invalid date'
  }
}

// Auto-select first session
onMounted(() => {
  if (chatSessions.value.length > 0) {
    activeSession.value = chatSessions.value[0]
  }
})
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>