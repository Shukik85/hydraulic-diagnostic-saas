<script setup lang="ts">
import { ref, computed, nextTick, watch } from 'vue';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface Props {
  systemId: string;
  systemName?: string;
}

const props = defineProps<Props>();

const { t } = useI18n();
const toast = useToast();

const messages = ref<Message[]>([
  {
    id: '1',
    role: 'assistant',
    content: t('diagnosis.chat.welcome'),
    timestamp: new Date(),
  },
]);

const inputMessage = ref('');
const isLoading = ref(false);
const chatContainer = ref<HTMLElement | null>(null);

const scrollToBottom = async (): Promise<void> => {
  await nextTick();
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
};

const sendMessage = async (): Promise<void> => {
  const message = inputMessage.value.trim();
  if (!message || isLoading.value) {
    return;
  }

  // Add user message
  const userMessage: Message = {
    id: `user-${Date.now()}`,
    role: 'user',
    content: message,
    timestamp: new Date(),
  };
  messages.value.push(userMessage);
  inputMessage.value = '';
  scrollToBottom();

  // Simulate AI response (replace with actual RAG API call)
  isLoading.value = true;
  const assistantMessage: Message = {
    id: `assistant-${Date.now()}`,
    role: 'assistant',
    content: '',
    timestamp: new Date(),
    isStreaming: true,
  };
  messages.value.push(assistantMessage);

  try {
    // Simulate streaming response
    const response = `Based on the sensor data from ${props.systemName || 'the hydraulic system'}, I can help you diagnose the issue. The pressure drop you're experiencing could be caused by:\n\n1. **Leakage in the system** - Check seals and connections\n2. **Pump performance degradation** - Monitor pump efficiency\n3. **Filter blockage** - Inspect and replace if necessary\n\nWould you like me to analyze the specific sensor readings?`;
    
    const words = response.split(' ');
    for (let i = 0; i < words.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 50));
      assistantMessage.content += (i > 0 ? ' ' : '') + words[i];
      scrollToBottom();
    }
    
    assistantMessage.isStreaming = false;
  } catch (error) {
    toast.error(t('diagnosis.chat.error'), 'AI Assistant');
    messages.value.pop();
  } finally {
    isLoading.value = false;
  }
};

const copyToClipboard = async (content: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(content);
    toast.success(t('common.copied'), '');
  } catch (error) {
    toast.error(t('common.copyFailed'), '');
  }
};

const formatContent = (content: string): string => {
  // Simple markdown formatting
  return content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
};
</script>

<template>
  <div class="flex h-full flex-col rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
    <!-- Header -->
    <div class="border-b border-gray-200 p-4 dark:border-gray-700">
      <div class="flex items-center gap-3">
        <div class="flex h-10 w-10 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900">
          <Icon name="heroicons:chat-bubble-left-right" class="h-6 w-6 text-primary-600 dark:text-primary-300" />
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            {{ t('diagnosis.chat.title') }}
          </h3>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ t('diagnosis.chat.subtitle') }}
          </p>
        </div>
      </div>
    </div>

    <!-- Messages -->
    <div
      ref="chatContainer"
      class="flex-1 space-y-4 overflow-y-auto p-4"
      role="log"
      aria-live="polite"
      aria-atomic="false"
    >
      <div
        v-for="message in messages"
        :key="message.id"
        class="flex"
        :class="{
          'justify-end': message.role === 'user',
          'justify-start': message.role === 'assistant',
        }"
      >
        <div
          class="group relative max-w-[80%] rounded-lg px-4 py-3"
          :class="{
            'bg-primary-600 text-white': message.role === 'user',
            'bg-gray-100 text-gray-900 dark:bg-gray-700 dark:text-gray-100': message.role === 'assistant',
          }"
        >
          <div
            class="prose prose-sm max-w-none"
            :class="{
              'prose-invert': message.role === 'user',
            }"
            v-html="formatContent(message.content)"
          />
          
          <!-- Streaming indicator -->
          <span
            v-if="message.isStreaming"
            class="ml-1 inline-block h-3 w-3 animate-pulse rounded-full bg-current"
            aria-label="Thinking..."
          />

          <!-- Timestamp & Actions -->
          <div class="mt-2 flex items-center justify-between text-xs opacity-70">
            <time :datetime="message.timestamp.toISOString()">
              {{ message.timestamp.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) }}
            </time>
            
            <button
              v-if="message.role === 'assistant' && !message.isStreaming"
              @click="copyToClipboard(message.content)"
              class="opacity-0 transition-opacity group-hover:opacity-100"
              :aria-label="t('common.copy')"
            >
              <Icon name="heroicons:clipboard-document" class="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="border-t border-gray-200 p-4 dark:border-gray-700">
      <form @submit.prevent="sendMessage" class="flex gap-2">
        <input
          v-model="inputMessage"
          type="text"
          :placeholder="t('diagnosis.chat.placeholder')"
          :disabled="isLoading"
          class="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          :aria-label="t('diagnosis.chat.inputLabel')"
        />
        <Button
          type="submit"
          variant="primary"
          :disabled="!inputMessage.trim() || isLoading"
          :loading="isLoading"
          :aria-label="t('common.send')"
        >
          <Icon name="heroicons:paper-airplane" class="h-5 w-5" />
        </Button>
      </form>
    </div>
  </div>
</template>

<style scoped>
.prose :deep(strong) {
  font-weight: 600;
}
</style>
