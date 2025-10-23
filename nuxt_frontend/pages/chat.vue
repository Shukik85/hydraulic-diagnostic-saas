<template>
  <div class="flex h-[calc(100vh-8rem)]">
    <!-- Chat Sidebar -->
    <div class="w-80 border-r bg-muted/20 flex flex-col">
      <div class="p-4 border-b">
        <h2 class="font-semibold">AI Assistant</h2>
        <p class="text-sm text-muted-foreground">RAG-powered hydraulic diagnostics</p>
      </div>

      <!-- Sessions List -->
      <div class="flex-1 overflow-y-auto p-2">
        <div class="space-y-2">
          <div
            v-for="session in chatSessions"
            :key="session.id"
            :class="`p-3 rounded-lg cursor-pointer transition-colors ${
              activeSession?.id === session.id ? 'bg-primary/10 border border-primary/20' : 'hover:bg-muted'
            }`"
            @click="selectSession(session)"
          >
            <p class="text-sm font-medium truncate">{{ session.title }}</p>
            <p class="text-xs text-muted-foreground">{{ session.lastMessage }}</p>
            <p class="text-xs text-muted-foreground mt-1">{{ session.timestamp }}</p>
          </div>
        </div>
      </div>

      <!-- New Chat Button -->
      <div class="p-4 border-t">
        <UiButton @click="startNewChat" class="w-full" variant="outline">
          <Icon name="lucide:plus" class="mr-2 h-4 w-4" />
          New Chat
        </UiButton>
      </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col">
      <!-- Chat Header -->
      <div class="p-4 border-b flex items-center justify-between">
        <div>
          <h3 class="font-medium">{{ activeSession?.title || 'New Chat' }}</h3>
          <p class="text-sm text-muted-foreground">{{ activeSession?.description || 'Ask me anything about your hydraulic systems' }}</p>
        </div>
        <div class="flex items-center gap-2">
          <UiButton variant="outline" size="sm">
            <Icon name="lucide:share" class="mr-2 h-4 w-4" />
            Share
          </UiButton>
          <UiButton variant="outline" size="sm">
            <Icon name="lucide:more-vertical" class="h-4 w-4" />
          </UiButton>
        </div>
      </div>

      <!-- Messages Area -->
      <div class="flex-1 overflow-y-auto p-4 space-y-4" ref="messagesContainer">
        <div
          v-for="message in activeSession?.messages || []"
          :key="message.id"
          :class="`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`"
        >
          <div
            :class="`max-w-[70%] p-3 rounded-lg ${
              message.role === 'user'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted'
            }`"
          >
            <div v-if="message.role === 'assistant'" class="space-y-2">
              <div v-html="formatMessage(message.content)"></div>
              <div v-if="message.sources && message.sources.length > 0" class="mt-3 pt-3 border-t border-muted-foreground/20">
                <p class="text-xs text-muted-foreground mb-2">Sources:</p>
                <div class="space-y-1">
                  <div
                    v-for="source in message.sources"
                    :key="source.id"
                    class="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  >
                    • {{ source.title }}
                  </div>
                </div>
              </div>
            </div>
            <div v-else>
              {{ message.content }}
            </div>
            <p class="text-xs opacity-70 mt-1">{{ message.timestamp }}</p>
          </div>
        </div>

        <!-- Typing Indicator -->
        <div v-if="isTyping" class="flex justify-start">
          <div class="bg-muted p-3 rounded-lg">
            <div class="flex space-x-1">
              <div class="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="p-4 border-t">
        <div class="flex space-x-2">
          <div class="flex-1 relative">
            <UiTextarea
              v-model="newMessage"
              placeholder="Ask about hydraulic diagnostics, maintenance, or system analysis..."
              class="min-h-[44px] max-h-32 resize-none pr-12"
              @keydown.enter.exact.prevent="sendMessage"
              @keydown.enter.shift.exact="newMessage += '\n'"
            />
            <div class="absolute right-3 bottom-3 flex items-center space-x-1">
              <UiButton
                variant="ghost"
                size="icon"
                class="h-6 w-6"
                @click="attachFile"
              >
                <Icon name="lucide:paperclip" class="h-4 w-4" />
              </UiButton>
            </div>
          </div>
          <UiButton @click="sendMessage" :disabled="!newMessage.trim() || isTyping">
            <Icon name="lucide:send" class="h-4 w-4" />
          </UiButton>
        </div>

        <!-- Context Panel Toggle -->
        <div class="flex items-center justify-between mt-2">
          <div class="flex items-center space-x-2">
            <UiButton
              variant="ghost"
              size="sm"
              @click="showContext = !showContext"
            >
              <Icon name="lucide:eye" class="mr-2 h-4 w-4" />
              Context
            </UiButton>
            <span class="text-xs text-muted-foreground">
              {{ contextItems.length }} items
            </span>
          </div>
          <div class="text-xs text-muted-foreground">
            Press Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>

    <!-- Context Panel -->
    <div
      v-if="showContext"
      class="w-80 border-l bg-muted/20 flex flex-col"
    >
      <div class="p-4 border-b">
        <h3 class="font-medium">Context</h3>
        <p class="text-sm text-muted-foreground">Relevant information for this conversation</p>
      </div>

      <div class="flex-1 overflow-y-auto p-4">
        <div class="space-y-3">
          <div
            v-for="item in contextItems"
            :key="item.id"
            class="p-3 bg-background rounded-lg border"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <p class="text-sm font-medium">{{ item.title }}</p>
                <p class="text-xs text-muted-foreground mt-1">{{ item.description }}</p>
                <p class="text-xs text-muted-foreground mt-1">{{ item.source }}</p>
              </div>
              <UiButton variant="ghost" size="icon" class="h-6 w-6">
                <Icon name="lucide:x" class="h-3 w-3" />
              </UiButton>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'

interface ChatMessage {
  id: number
  role: string
  content: string
  timestamp: string
  sources?: { id: number; title: string }[]
}

interface ChatSession {
  id: number
  title: string
  description: string
  lastMessage: string
  timestamp: string
  messages: ChatMessage[]
}

const activeSession = ref<ChatSession | null>(null)
const newMessage = ref('')
const isTyping = ref(false)
const showContext = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)

const chatSessions = ref([
  {
    id: 1,
    title: 'Pressure System Analysis',
    description: 'Discussion about HYD-001 pressure issues',
    lastMessage: 'What are the recommended maintenance intervals...',
    timestamp: '2 hours ago',
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'I\'m seeing pressure fluctuations in HYD-001. What could be causing this?',
        timestamp: '14:30'
      },
      {
        id: 2,
        role: 'assistant',
        content: 'Pressure fluctuations in hydraulic systems can be caused by several factors. Let me analyze the most common causes based on your system data:\n\n1. **Air entrainment** - Air bubbles in the fluid can cause pressure variations\n2. **Pump cavitation** - When the pump inlet pressure is too low\n3. **Filter clogging** - Restricted flow through dirty filters\n4. **Valve issues** - Malfunctioning relief or control valves\n\nBased on your recent sensor data, I notice the pressure variations correlate with temperature changes, which suggests thermal expansion of the fluid might be a factor.',
        timestamp: '14:31',
        sources: [
          { id: 1, title: 'Hydraulic System Troubleshooting Guide' },
          { id: 2, title: 'Pressure Control Best Practices' }
        ]
      },
      {
        id: 3,
        role: 'user',
        content: 'What are the recommended maintenance intervals for the filters?',
        timestamp: '14:32'
      }
    ]
  },
  {
    id: 2,
    title: 'Vibration Analysis Help',
    description: 'Questions about vibration monitoring',
    lastMessage: 'How do I interpret vibration frequency...',
    timestamp: '1 day ago',
    messages: []
  }
])

const contextItems = ref([
  {
    id: 1,
    title: 'HYD-001 System Manual',
    description: 'Complete operation and maintenance guide for Pump Station A',
    source: 'Documentation'
  },
  {
    id: 2,
    title: 'Recent Pressure Data',
    description: 'Sensor readings from the last 24 hours showing pressure variations',
    source: 'Live Data'
  },
  {
    id: 3,
    title: 'Maintenance Schedule',
    description: 'Upcoming service tasks and filter replacement dates',
    source: 'System Records'
  }
])

const selectSession = (session: ChatSession) => {
  activeSession.value = session
  nextTick(() => {
    scrollToBottom()
  })
}

const startNewChat = () => {
  const newSession = {
    id: Date.now(),
    title: 'New Chat',
    description: 'Ask me anything about your hydraulic systems',
    lastMessage: '',
    timestamp: 'Just now',
    messages: []
  }
  chatSessions.value.unshift(newSession)
  activeSession.value = newSession
}

const sendMessage = async () => {
  if (!newMessage.value.trim()) return

  const message = {
    id: Date.now(),
    role: 'user',
    content: newMessage.value,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  if (!activeSession.value) {
    startNewChat()
  }

  if (activeSession.value) {
    activeSession.value.messages.push(message)
    activeSession.value.lastMessage = message.content.slice(0, 50) + '...'
  }

  newMessage.value = ''
  isTyping.value = true

  nextTick(() => {
    scrollToBottom()
  })

  // Simulate AI response
  setTimeout(() => {
    const aiMessage = {
      id: Date.now(),
      role: 'assistant',
      content: 'I\'m analyzing your question about hydraulic system diagnostics. Based on the information available, here\'s what I can tell you:\n\n**System Health Overview**\n- Current pressure: 145 PSI (within normal range)\n- Temperature: 68°C (optimal)\n- Flow rate: 22 L/min (stable)\n\n**Recommendations**\n1. Check filter condition - due for replacement\n2. Monitor vibration levels - slight increase detected\n3. Schedule preventive maintenance\n\nWould you like me to run a detailed diagnostic or provide more specific guidance?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      sources: [
        { id: 1, title: 'System Health Report' },
        { id: 2, title: 'Maintenance Guidelines' }
      ]
    }

    if (activeSession.value) {
      activeSession.value.messages.push(aiMessage)
    }
    isTyping.value = false

    nextTick(() => {
      scrollToBottom()
    })
  }, 2000)
}

const attachFile = () => {
  // File attachment functionality would go here
  console.log('Attach file clicked')
}

const formatMessage = (content: string) => {
  // Simple markdown-like formatting
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
}

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

onMounted(() => {
  if (chatSessions.value.length > 0) {
    activeSession.value = chatSessions.value[0]
  }
})
</script>
