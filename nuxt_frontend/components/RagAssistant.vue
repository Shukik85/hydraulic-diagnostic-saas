<template>
  <div class="rag-assistant">
    <div class="chat-header">
      <h3>RAG Assistant</h3>
    </div>

    <div class="messages-container" ref="messagesContainer">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['message', message.role]"
      >
        <div class="message-content">{{ message.content }}</div>
      </div>
    </div>

    <div class="input-container">
      <input
        v-model="query"
        @keyup.enter="sendMessage"
        placeholder="Введите ваш запрос..."
        class="query-input"
        :disabled="isLoading"
      />
      <button
        @click="sendMessage"
        :disabled="isLoading || !query.trim()"
        class="send-button"
      >
        {{ isLoading ? 'Отправка...' : 'Отправить' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { useRagAssistant } from '~/composables/useRagAssistant'

const { messages, isLoading, sendQuery } = useRagAssistant()
const query = ref('')
const messagesContainer = ref(null)

const sendMessage = async () => {
  if (!query.value.trim() || isLoading.value) return

  const userQuery = query.value
  query.value = ''

  await sendQuery(userQuery)

  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.rag-assistant {
  display: flex;
  flex-direction: column;
  height: 600px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: white;
}

.chat-header {
  padding: 1rem;
  border-bottom: 1px solid #ddd;
  background: #f5f5f5;
}

.chat-header h3 {
  margin: 0;
  font-size: 1.25rem;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.message {
  max-width: 80%;
  padding: 0.75rem;
  border-radius: 8px;
  word-wrap: break-word;
}

.message.user {
  align-self: flex-end;
  background: #007bff;
  color: white;
}

.message.assistant {
  align-self: flex-start;
  background: #f0f0f0;
  color: #333;
}

.message-content {
  white-space: pre-wrap;
}

.input-container {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  border-top: 1px solid #ddd;
}

.query-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
}

.query-input:focus {
  outline: none;
  border-color: #007bff;
}

.send-button {
  padding: 0.75rem 1.5rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  transition: background 0.2s;
}

.send-button:hover:not(:disabled) {
  background: #0056b3;
}

.send-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
