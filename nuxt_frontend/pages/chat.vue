<script setup lang="ts">
// Fixed chat page with proper TypeScript types
import type { ChatSession, ChatMessage } from '~/types/api';

definePageMeta({
  middleware: 'auth',
});

useSeoMeta({
  title: 'AI Чат | Hydraulic Diagnostic SaaS',
  description: 'Интеллектуальный помощник для диагностики гидравлических систем',
});

const authStore = useAuthStore();

// Chat state with proper types
const activeSession = ref<ChatSession | null>(null);
const newMessage = ref<string>('');
const isLoading = ref<boolean>(false);
const showNewSessionModal = ref<boolean>(false);
const newSessionTitle = ref<string>('');

// Demo chat sessions with full type coverage
const chatSessions = ref<ChatSession[]>([
  {
    id: 1,
    title: 'Диагностика HYD-001',
    description: 'Анализ системы охлаждения',
    lastMessage: 'Система работает стабильно, но рекомендую...',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'Проанализируй состояние системы HYD-001',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: 2,
        role: 'assistant',
        content:
          'Система HYD-001 работает стабильно. Температура: 45.2°C, давление: 150.8 бар. Рекомендую проверить фильтры в течение 2 недель.',
        timestamp: new Date(Date.now() - 3580000).toISOString(),
        sources: [
          { title: 'Технические спецификации HYD-001', url: '/docs/hyd-001-specs.pdf' },
          { title: 'История обслуживания', url: '/maintenance/hyd-001-history' },
        ],
      },
    ],
  },
  {
    id: 2,
    title: 'Оптимизация энергопотребления',
    description: 'Консультация по снижению расходов',
    lastMessage: 'Рекомендую установить частотные преобразователи...',
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'Как снизить энергопотребление гидравлических систем?',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: 2,
        role: 'assistant',
        content:
          'Рекомендую установить частотные преобразователи с регулируемой скоростью и систему рекуперации энергии.',
        timestamp: new Date(Date.now() - 7180000).toISOString(),
      },
    ],
  },
]);

// Lifecycle - Fixed type assignment
onMounted(() => {
  if (chatSessions.value.length > 0) {
    const firstSession = chatSessions.value[0];
    if (firstSession) {
      activeSession.value = firstSession;
    }
  }
});

// Chat methods
const selectSession = (session: ChatSession): void => {
  activeSession.value = session;
};

const sendMessage = async (): Promise<void> => {
  if (!newMessage.value.trim() || !activeSession.value || isLoading.value) return;

  const message = newMessage.value.trim();
  newMessage.value = '';
  isLoading.value = true;

  // Add user message
  const userMessage: ChatMessage = {
    id: Date.now(),
    role: 'user',
    content: message,
    timestamp: new Date().toISOString(),
  };

  activeSession.value.messages.push(userMessage);

  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Add assistant response
    const assistantMessage: ChatMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: `Ответ на: "${message}". Я проанализировал ваш запрос и могу помочь с диагностикой гидравлической системы.`,
      timestamp: new Date().toISOString(),
      sources: [{ title: 'Техническая документация', url: '/docs/hydraulic-systems.pdf' }],
    };

    if (activeSession.value) {
      activeSession.value.messages.push(assistantMessage);
      activeSession.value.lastMessage = assistantMessage.content.substring(0, 100) + '...';
      activeSession.value.timestamp = assistantMessage.timestamp;
    }
  } catch (error) {
    console.error('Chat error:', error);
  } finally {
    isLoading.value = false;
  }
};

const createNewSession = (): void => {
  if (!newSessionTitle.value.trim()) return;

  const newSession: ChatSession = {
    id: Date.now(),
    title: newSessionTitle.value.trim(),
    description: 'Новая консультация',
    lastMessage: '',
    timestamp: new Date().toISOString(),
    messages: [],
  };

  chatSessions.value.unshift(newSession);
  activeSession.value = newSession;
  newSessionTitle.value = '';
  showNewSessionModal.value = false;
};

const formatTimestamp = (timestamp: string): string => {
  try {
    return new Date(timestamp).toLocaleString('ru-RU', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return 'Неверная дата';
  }
};
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <div class="flex h-screen">
      <!-- Sidebar: Chat Sessions -->
      <div
        class="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col"
      >
        <!-- Header -->
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between mb-4">
            <h2 class="premium-heading-sm text-gray-900 dark:text-white">🤖 AI Чат</h2>
            <PremiumButton size="sm" icon="heroicons:plus" @click="showNewSessionModal = true">
              Новый
            </PremiumButton>
          </div>
          <p class="text-sm text-gray-600 dark:text-gray-300">
            Интеллектуальный помощник для диагностики
          </p>
        </div>

        <!-- Sessions list -->
        <div class="flex-1 overflow-y-auto p-4 space-y-3">
          <div
            v-for="session in chatSessions"
            :key="session.id"
            @click="selectSession(session)"
            :class="[
              'p-4 rounded-lg cursor-pointer transition-all',
              activeSession?.id === session.id
                ? 'bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700'
                : 'hover:bg-gray-50 dark:hover:bg-gray-700 border border-transparent',
            ]"
          >
            <h3 class="font-medium text-gray-900 dark:text-white mb-1 truncate">
              {{ session.title }}
            </h3>
            <p class="text-sm text-gray-500 dark:text-gray-400 mb-2 line-clamp-2">
              {{ session.lastMessage || session.description }}
            </p>
            <div class="flex items-center justify-between">
              <span class="text-xs text-gray-400 dark:text-gray-500">
                {{ formatTimestamp(session.timestamp) }}
              </span>
              <div class="flex items-center space-x-1">
                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                <span class="text-xs text-gray-400 dark:text-gray-500">{{
                  session.messages.length
                }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Main Chat Area -->
      <div class="flex-1 flex flex-col">
        <!-- Chat Header -->
        <div
          v-if="activeSession"
          class="p-6 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800"
        >
          <h1 class="premium-heading-md text-gray-900 dark:text-white mb-1">
            {{ activeSession.title }}
          </h1>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            {{ activeSession.description }}
          </p>
        </div>

        <!-- Messages -->
        <div class="flex-1 overflow-y-auto p-6 space-y-6">
          <div v-if="activeSession" class="max-w-4xl mx-auto">
            <div
              v-for="message in activeSession.messages"
              :key="message.id"
              :class="[
                'flex items-start space-x-4',
                message.role === 'user' ? 'justify-end' : 'justify-start',
              ]"
            >
              <!-- Avatar -->
              <div v-if="message.role === 'assistant'" class="flex-shrink-0">
                <div
                  class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center"
                >
                  <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
                </div>
              </div>

              <!-- Message content -->
              <div
                :class="[
                  'max-w-2xl',
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-l-2xl rounded-tr-2xl p-4'
                    : 'bg-white dark:bg-gray-800 rounded-r-2xl rounded-tl-2xl p-4 shadow-md border border-gray-200 dark:border-gray-700',
                ]"
              >
                <p
                  class="text-sm leading-relaxed whitespace-pre-wrap"
                  :class="message.role === 'user' ? 'text-white' : 'text-gray-900 dark:text-white'"
                >
                  {{ message.content }}
                </p>

                <!-- Sources -->
                <div
                  v-if="message.sources && message.sources.length > 0"
                  class="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600"
                >
                  <p class="text-xs text-gray-500 dark:text-gray-400 mb-2">Источники:</p>
                  <div class="space-y-1">
                    <a
                      v-for="source in message.sources"
                      :key="source.url"
                      :href="source.url"
                      target="_blank"
                      rel="noopener noreferrer"
                      class="block text-xs text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      {{ source.title }}
                    </a>
                  </div>
                </div>

                <p class="text-xs mt-2 opacity-70">
                  {{ formatTimestamp(message.timestamp) }}
                </p>
              </div>

              <!-- User avatar -->
              <div v-if="message.role === 'user'" class="flex-shrink-0">
                <div
                  class="w-8 h-8 bg-gradient-to-br from-gray-500 to-gray-600 rounded-full flex items-center justify-center"
                >
                  <Icon name="heroicons:user" class="w-4 h-4 text-white" />
                </div>
              </div>
            </div>
          </div>

          <!-- Empty state -->
          <div v-else class="flex-1 flex items-center justify-center">
            <div class="text-center max-w-md">
              <Icon
                name="heroicons:chat-bubble-left-ellipsis"
                class="w-16 h-16 mx-auto text-gray-400 mb-4"
              />
              <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Выберите чат или создайте новый
              </h3>
              <p class="text-gray-500 dark:text-gray-400 mb-6">
                AI поможет вам с диагностикой и оптимизацией систем
              </p>
              <PremiumButton @click="showNewSessionModal = true" icon="heroicons:plus">
                Создать новый чат
              </PremiumButton>
            </div>
          </div>
        </div>

        <!-- Message input -->
        <div
          v-if="activeSession"
          class="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6"
        >
          <form @submit.prevent="sendMessage" class="flex items-end space-x-4">
            <div class="flex-1">
              <textarea
                v-model="newMessage"
                :disabled="isLoading"
                placeholder="Задайте вопрос о диагностике..."
                rows="3"
                class="premium-input resize-none"
                @keydown.meta.enter.prevent="sendMessage"
                @keydown.ctrl.enter.prevent="sendMessage"
              ></textarea>
            </div>
            <PremiumButton
              type="submit"
              :disabled="!newMessage.trim() || isLoading"
              :loading="isLoading"
              icon="heroicons:paper-airplane"
              gradient
            >
              Отправить
            </PremiumButton>
          </form>
          <p class="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Нажмите Cmd+Enter для отправки
          </p>
        </div>
      </div>
    </div>

    <!-- New Session Modal -->
    <div
      v-if="showNewSessionModal"
      class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
      @click="showNewSessionModal = false"
    >
      <div class="premium-card max-w-md w-full m-4" @click.stop>
        <div class="p-6 border-b border-gray-200 dark:border-gray-700">
          <h3 class="premium-heading-sm text-gray-900 dark:text-white">🆕 Новая консультация</h3>
          <p class="premium-body text-gray-600 dark:text-gray-300">Опишите вашу задачу</p>
        </div>

        <form @submit.prevent="createNewSession" class="p-6 space-y-4">
          <div>
            <label for="sessionTitle" class="premium-label">Тема консультации</label>
            <input
              id="sessionTitle"
              v-model="newSessionTitle"
              type="text"
              required
              class="premium-input"
              placeholder="напр.: Оптимизация HYD-002"
              autofocus
            />
          </div>

          <div class="flex space-x-3">
            <PremiumButton
              type="button"
              variant="secondary"
              @click="showNewSessionModal = false"
              class="flex-1"
            >
              Отмена
            </PremiumButton>
            <PremiumButton
              type="submit"
              :disabled="!newSessionTitle.trim()"
              icon="heroicons:plus"
              class="flex-1"
            >
              Создать
            </PremiumButton>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
