<template>
  <div class="ai-chat-overlay" @click.self="$emit('close')">
    <div class="ai-chat-container">
      <!-- Заголовок чата -->
      <div class="chat-header">
        <div class="chat-title">
          <div class="chat-icon">🤖</div>
          <div class="chat-info">
            <h3>AI Помощник по гидравлике</h3>
            <p class="chat-status">{{ connectionStatus }}</p>
          </div>
        </div>
        
        <div class="chat-controls">
          <button class="control-btn" @click="clearChat" title="Очистить чат">
            🗑️
          </button>
          <button class="control-btn" @click="exportChat" title="Экспорт чата">
            💾
          </button>
          <button class="control-btn close-btn" @click="$emit('close')" title="Закрыть">
            ✕
          </button>
        </div>
      </div>

      <!-- Область сообщений -->
      <div class="chat-messages" ref="messagesContainer">
        <!-- Приветственное сообщение -->
        <div v-if="messages.length === 0" class="welcome-message">
          <div class="welcome-icon">👋</div>
          <div class="welcome-text">
            <h4>Добро пожаловать!</h4>
            <p>Я AI-помощник по диагностике гидравлических систем. Могу помочь с:</p>
            <ul>
              <li>🔍 Анализом проблем в системах</li>
              <li>📚 Поиском по базе знаний ГОСТ</li>
              <li>💡 Рекомендациями по обслуживанию</li>
              <li>📊 Интерпретацией данных датчиков</li>
            </ul>
          </div>
          
          <div class="quick-questions">
            <p>Быстрые вопросы:</p>
            <div class="quick-question-buttons">
              <button 
                v-for="question in quickQuestions" 
                :key="question.id"
                class="quick-question-btn"
                @click="sendQuickQuestion(question.text)"
              >
                {{ question.icon }} {{ question.text }}
              </button>
            </div>
          </div>
        </div>

        <!-- Сообщения чата -->
        <div 
          v-for="message in messages" 
          :key="message.id"
          class="message"
          :class="{ 'user-message': message.isUser, 'ai-message': !message.isUser }"
        >
          <div class="message-avatar">
            {{ message.isUser ? '👤' : '🤖' }}
          </div>
          
          <div class="message-content">
            <div class="message-header">
              <span class="message-sender">
                {{ message.isUser ? 'Вы' : 'AI Помощник' }}
              </span>
              <span class="message-time">
                {{ formatTime(message.timestamp) }}
              </span>
            </div>
            
            <div class="message-text" v-html="formatMessage(message.text)"></div>
            
            <!-- Источники для AI ответов -->
            <div v-if="!message.isUser && message.sources" class="message-sources">
              <details class="sources-details">
                <summary>📚 Источники ({{ message.sources.length }})</summary>
                <div class="sources-list">
                  <div 
                    v-for="source in message.sources" 
                    :key="source.title"
                    class="source-item"
                  >
                    <div class="source-title">{{ source.title }}</div>
                    <div class="source-relevance">
                      Релевантность: {{ Math.round(source.relevance * 100) }}%
                    </div>
                  </div>
                </div>
              </details>
            </div>
            
            <!-- Рекомендации -->
            <div v-if="!message.isUser && message.recommendations" class="message-recommendations">
              <h5>💡 Рекомендации:</h5>
              <div class="recommendations-list">
                <div 
                  v-for="rec in message.recommendations" 
                  :key="rec.id"
                  class="recommendation-item"
                  :class="`priority-${rec.priority}`"
                >
                  <div class="recommendation-icon">{{ getPriorityIcon(rec.priority) }}</div>
                  <div class="recommendation-content">
                    <div class="recommendation-title">{{ rec.title }}</div>
                    <div class="recommendation-description">{{ rec.description }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Действия сообщения -->
          <div class="message-actions">
            <button 
              class="action-btn" 
              @click="copyMessage(message.text)"
              title="Копировать"
            >
              📋
            </button>
            <button 
              v-if="!message.isUser" 
              class="action-btn" 
              @click="rateMessage(message, 'like')"
              :class="{ active: message.rating === 'like' }"
              title="Полезно"
            >
              👍
            </button>
            <button 
              v-if="!message.isUser" 
              class="action-btn" 
              @click="rateMessage(message, 'dislike')"
              :class="{ active: message.rating === 'dislike' }"
              title="Не полезно"
            >
              👎
            </button>
          </div>
        </div>

        <!-- Индикатор печатания -->
        <div v-if="isTyping" class="typing-indicator">
          <div class="message ai-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
              <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <div class="typing-text">AI анализирует ваш вопрос...</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Панель ввода -->
      <div class="chat-input-panel">
        <!-- Контекстные предложения -->
        <div v-if="contextSuggestions.length > 0" class="context-suggestions">
          <div class="suggestions-label">💡 Возможно, вас интересует:</div>
          <div class="suggestions-list">
            <button 
              v-for="suggestion in contextSuggestions" 
              :key="suggestion.id"
              class="suggestion-btn"
              @click="sendMessage(suggestion.text)"
            >
              {{ suggestion.text }}
            </button>
          </div>
        </div>

        <!-- Поле ввода -->
        <div class="input-container">
          <div class="input-wrapper">
            <textarea
              v-model="inputMessage"
              @keydown="handleKeyDown"
              placeholder="Задайте вопрос о гидравлических системах..."
              class="message-input"
              rows="1"
              ref="messageInput"
              :disabled="isTyping"
            ></textarea>
            
            <div class="input-actions">
              <!-- Прикрепление файла -->
              <button 
                class="attachment-btn" 
                @click="$refs.fileInput.click()"
                title="Прикрепить данные датчиков"
              >
                📎
              </button>
              <input 
                type="file" 
                ref="fileInput" 
                @change="handleFileUpload"
                accept=".csv,.json,.txt"
                style="display: none"
              >
              
              <!-- Голосовой ввод (заглушка) -->
              <button 
                class="voice-btn" 
                @click="startVoiceInput"
                :class="{ active: isListening }"
                title="Голосовой ввод"
              >
                🎤
              </button>
              
              <!-- Отправка сообщения -->
              <button 
                class="send-btn" 
                @click="sendMessage()"
                :disabled="!inputMessage.trim() || isTyping"
                title="Отправить (Ctrl+Enter)"
              >
                <span v-if="isTyping" class="sending-icon">⏳</span>
                <span v-else>📤</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Счетчик символов и статус -->
        <div class="input-footer">
          <div class="character-count">
            {{ inputMessage.length }}/1000
          </div>
          <div class="ai-status">
            <span class="status-indicator" :class="aiStatusClass"></span>
            {{ aiStatusText }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { ragService } from '@/services/ragService'

export default {
  name: 'AIChat',
  emits: ['close'],
  setup(props, { emit }) {
    // Реактивные данные
    const messages = ref([])
    const inputMessage = ref('')
    const isTyping = ref(false)
    const isListening = ref(false)
    const connectionStatus = ref('Подключен')
    const contextSuggestions = ref([])
    const messagesContainer = ref(null)
    const messageInput = ref(null)
    const fileInput = ref(null)
    
    const quickQuestions = [
      {
        id: 1,
        icon: '🔧',
        text: 'Что означает высокое давление в системе?'
      },
      {
        id: 2,
        icon: '🌡️',
        text: 'Нормальная температура масла в гидросистеме?'
      },
      {
        id: 3,
        icon: '⚠️',
        text: 'Причины вибрации в гидронасосе?'
      },
      {
        id: 4,
        icon: '🔍',
        text: 'Как диагностировать утечки в системе?'
      }
    ]
    
    // Вычисляемые свойства
    const aiStatusClass = computed(() => {
      if (isTyping.value) return 'thinking'
      if (connectionStatus.value === 'Подключен') return 'online'
      return 'offline'
    })
    
    const aiStatusText = computed(() => {
      if (isTyping.value) return 'Анализирует...'
      return connectionStatus.value
    })
    
    // Методы
    const sendMessage = async (text = null) => {
      const messageText = text || inputMessage.value.trim()
      if (!messageText || isTyping.value) return
      
      // Добавляем сообщение пользователя
      const userMessage = {
        id: Date.now(),
        text: messageText,
        isUser: true,
        timestamp: new Date()
      }
      
      messages.value.push(userMessage)
      
      if (!text) {
        inputMessage.value = ''
      }
      
      // Автоскролл
      await nextTick()
      scrollToBottom()
      
      // Получаем ответ от AI
      await getAIResponse(messageText)
    }
    
    const getAIResponse = async (question) => {
      isTyping.value = true
      
      try {
        // Имитируем задержку для реалистичности
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        // Запрос к RAG системе
        const response = await ragService.askQuestion(question)
        
        // Создаем сообщение AI
        const aiMessage = {
          id: Date.now(),
          text: response.answer,
          isUser: false,
          timestamp: new Date(),
          sources: response.sources || [],
          confidence: response.confidence || 0
        }
        
        // Если есть рекомендации, добавляем их
        if (response.sources && response.sources.length > 0) {
          const recommendations = await generateRecommendations(response.sources)
          aiMessage.recommendations = recommendations
        }
        
        messages.value.push(aiMessage)
        
        // Генерируем контекстные предложения
        generateContextSuggestions(question, response.answer)
        
      } catch (error) {
        console.error('Ошибка получения ответа AI:', error)
        
        // Добавляем сообщение об ошибке
        const errorMessage = {
          id: Date.now(),
          text: 'Извините, произошла ошибка при обработке вашего вопроса. Попробуйте переформулировать или задать другой вопрос.',
          isUser: false,
          timestamp: new Date(),
          isError: true
        }
        
        messages.value.push(errorMessage)
      } finally {
        isTyping.value = false
        await nextTick()
        scrollToBottom()
      }
    }
    
    const generateRecommendations = async (sources) => {
      // Простая логика генерации рекомендаций на основе источников
      const recommendations = []
      
      sources.forEach((source, index) => {
        if (source.category === 'diagnostics' && source.relevance > 0.7) {
          recommendations.push({
            id: index + 1,
            title: `Проверка на основе ${source.title}`,
            description: 'Рекомендуется выполнить диагностику по данному направлению',
            priority: source.relevance > 0.8 ? 'high' : 'medium'
          })
        }
      })
      
      return recommendations.slice(0, 3) // Максимум 3 рекомендации
    }
    
    const generateContextSuggestions = (question, answer) => {
      // Генерация контекстных предложений на основе вопроса и ответа
      const suggestions = []
      
      if (question.toLowerCase().includes('давление')) {
        suggestions.push(
          { id: 1, text: 'Какие нормы давления по ГОСТ?' },
          { id: 2, text: 'Как измерить давление точно?' }
        )
      }
      
      if (question.toLowerCase().includes('температура')) {
        suggestions.push(
          { id: 3, text: 'Влияние температуры на вязкость масла' },
          { id: 4, text: 'Системы охлаждения гидромасла' }
        )
      }
      
      if (answer.includes('фильтр')) {
        suggestions.push(
          { id: 5, text: 'Периодичность замены фильтров' }
        )
      }
      
      contextSuggestions.value = suggestions.slice(0, 2)
      
      // Убираем предложения через некоторое время
      setTimeout(() => {
        contextSuggestions.value = []
      }, 15000)
    }
    
    const sendQuickQuestion = (question) => {
      sendMessage(question)
    }
    
    const handleKeyDown = (event) => {
      if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault()
        sendMessage()
      } else if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        sendMessage()
      }
    }
    
    const handleFileUpload = async (event) => {
      const file = event.target.files[0]
      if (!file) return
      
      // Простая обработка файла
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target.result
        const fileName = file.name
        
        // Добавляем сообщение о загрузке файла
        const fileMessage = {
          id: Date.now(),
          text: `📎 Загружен файл: ${fileName}\n\nАнализирую содержимое...`,
          isUser: true,
          timestamp: new Date()
        }
        
        messages.value.push(fileMessage)
        
        // Имитируем анализ файла
        setTimeout(() => {
          const analysisText = `Проанализировал файл ${fileName}. ` +
            `Найдено ${Math.floor(Math.random() * 100)} записей данных датчиков. ` +
            `Есть вопросы по этим данным?`
          
          sendMessage(analysisText)
        }, 2000)
      }
      
      reader.readAsText(file)
      event.target.value = '' // Сброс input
    }
    
    const startVoiceInput = () => {
      // Заглушка для голосового ввода
      isListening.value = !isListening.value
      
      if (isListening.value) {
        // Имитируем голосовое распознавание
        setTimeout(() => {
          inputMessage.value = 'Пример голосового вопроса: какая температура считается критической?'
          isListening.value = false
        }, 3000)
      }
    }
    
    const clearChat = () => {
      if (confirm('Очистить историю чата?')) {
        messages.value = []
        contextSuggestions.value = []
      }
    }
    
    const exportChat = () => {
      const chatData = {
        messages: messages.value.map(msg => ({
          sender: msg.isUser ? 'User' : 'AI',
          text: msg.text,
          timestamp: msg.timestamp,
          sources: msg.sources || []
        })),
        exportedAt: new Date().toISOString()
      }
      
      const dataStr = JSON.stringify(chatData, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      
      const link = document.createElement('a')
      link.href = URL.createObjectURL(dataBlob)
      link.download = `ai_chat_${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    
    const copyMessage = async (text) => {
      try {
        await navigator.clipboard.writeText(text)
        // Можно добавить уведомление об успешном копировании
      } catch (error) {
        console.error('Ошибка копирования:', error)
      }
    }
    
    const rateMessage = (message, rating) => {
      message.rating = message.rating === rating ? null : rating
      
      // Отправка рейтинга на сервер (заглушка)
      console.log('Сообщение оценено:', message.id, rating)
    }
    
    const scrollToBottom = () => {
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      }
    }
    
    const formatTime = (date) => {
      return date.toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    
    const formatMessage = (text) => {
      // Простое форматирование текста
      return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>')
    }
    
    const getPriorityIcon = (priority) => {
      const icons = {
        'urgent': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
      }
      return icons[priority] || '🟡'
    }
    
    // Автоматическое изменение размера textarea
    const autoResizeTextarea = () => {
      const textarea = messageInput.value
      if (textarea) {
        textarea.style.height = 'auto'
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
      }
    }
    
    // Наблюдатели
    watch(inputMessage, () => {
      nextTick(() => {
        autoResizeTextarea()
      })
    })
    
    // Жизненный цикл
    onMounted(() => {
      // Фокус на поле ввода
      nextTick(() => {
        if (messageInput.value) {
          messageInput.value.focus()
        }
      })
    })
    
    onUnmounted(() => {
      // Очистка таймеров если есть
    })
    
    return {
      messages,
      inputMessage,
      isTyping,
      isListening,
      connectionStatus,
      contextSuggestions,
      messagesContainer,
      messageInput,
      fileInput,
      quickQuestions,
      aiStatusClass,
      aiStatusText,
      sendMessage,
      sendQuickQuestion,
      handleKeyDown,
      handleFileUpload,
      startVoiceInput,
      clearChat,
      exportChat,
      copyMessage,
      rateMessage,
      formatTime,
      formatMessage,
      getPriorityIcon
    }
  }
}
</script>

<style scoped>
/* Основные стили AI Chat */
.ai-chat-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 1rem;
}

.ai-chat-container {
  background: white;
  border-radius: 20px;
  width: 100%;
  max-width: 800px;
  height: 80vh;
  max-height: 700px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

/* Заголовок чата */
.chat-header {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 1.5rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-title {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.chat-icon {
  font-size: 2rem;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chat-info h3 {
  margin: 0 0 0.25rem 0;
  font-size: 1.25rem;
}

.chat-status {
  margin: 0;
  opacity: 0.8;
  font-size: 0.875rem;
}

.chat-controls {
  display: flex;
  gap: 0.5rem;
}

.control-btn {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  padding: 0.5rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 1.125rem;
}

.control-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* Область сообщений */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #f8fafc;
}

.welcome-message {
  text-align: center;
  padding: 3rem 2rem;
  color: #4a5568;
}

.welcome-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.welcome-text h4 {
  color: #2d3748;
  margin: 0 0 1rem 0;
  font-size: 1.5rem;
}

.welcome-text p {
  margin: 0 0 1rem 0;
}

.welcome-text ul {
  text-align: left;
  display: inline-block;
  margin: 0 0 2rem 0;
}

.quick-questions p {
  margin: 0 0 1rem 0;
  font-weight: 600;
  color: #2d3748;
}

.quick-question-buttons {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
  max-width: 600px;
  margin: 0 auto;
}

.quick-question-btn {
  background: white;
  border: 2px solid #e2e8f0;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  font-size: 0.875rem;
}

.quick-question-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
  transform: translateY(-2px);
}

/* Сообщения */
.message {
  display: flex;
  margin-bottom: 1.5rem;
  animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.user-message {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  margin: 0 1rem;
  flex-shrink: 0;
}

.user-message .message-avatar {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.ai-message .message-avatar {
  background: linear-gradient(135deg, #48bb78, #38a169);
}

.message-content {
  flex: 1;
  max-width: 70%;
}

.user-message .message-content {
  text-align: right;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.user-message .message-header {
  flex-direction: row-reverse;
}

.message-sender {
  font-weight: 600;
  color: #2d3748;
  font-size: 0.875rem;
}

.message-time {
  color: #a0aec0;
  font-size: 0.75rem;
}

.message-text {
  background: white;
  padding: 1rem;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  line-height: 1.6;
  word-wrap: break-word;
}

.user-message .message-text {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

/* Источники */
.message-sources {
  margin-top: 1rem;
}

.sources-details {
  background: #f0f4f8;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.sources-details summary {
  padding: 0.75rem;
  cursor: pointer;
  font-weight: 500;
  color: #4a5568;
}

.sources-list {
  padding: 0 0.75rem 0.75rem;
}

.source-item {
  background: white;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 6px;
  border-left: 3px solid #667eea;
}

.source-title {
  font-weight: 500;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.source-relevance {
  color: #718096;
  font-size: 0.8rem;
}

/* Рекомендации */
.message-recommendations {
  margin-top: 1rem;
  background: #f0fff4;
  border-radius: 8px;
  padding: 1rem;
  border: 1px solid #c6f6d5;
}

.message-recommendations h5 {
  margin: 0 0 0.75rem 0;
  color: #2d3748;
}

.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.recommendation-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  background: white;
  padding: 0.75rem;
  border-radius: 8px;
  border-left: 3px solid;
}

.priority-urgent { border-left-color: #f56565; }
.priority-high { border-left-color: #ed8936; }
.priority-medium { border-left-color: #f6e05e; }
.priority-low { border-left-color: #68d391; }

.recommendation-icon {
  font-size: 1.25rem;
}

.recommendation-content {
  flex: 1;
}

.recommendation-title {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.recommendation-description {
  color: #4a5568;
  font-size: 0.875rem;
}

/* Действия сообщения */
.message-actions {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin: 0 1rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.message:hover .message-actions {
  opacity: 1;
}

.action-btn {
  background: #e2e8f0;
  border: none;
  padding: 0.375rem;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.875rem;
}

.action-btn:hover {
  background: #cbd5e0;
}

.action-btn.active {
  background: #667eea;
  color: white;
}

/* Индикатор печатания */
.typing-indicator {
  margin-bottom: 1.5rem;
}

.typing-animation {
  display: flex;
  gap: 0.25rem;
  margin-bottom: 0.5rem;
}

.typing-animation span {
  width: 6px;
  height: 6px;
  background: #cbd5e0;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-animation span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-animation span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.typing-text {
  color: #718096;
  font-style: italic;
  font-size: 0.875rem;
}

/* Панель ввода */
.chat-input-panel {
  background: white;
  border-top: 1px solid #e2e8f0;
  padding: 1rem;
}

.context-suggestions {
  margin-bottom: 1rem;
  background: #f0f4f8;
  padding: 1rem;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

.suggestions-label {
  font-size: 0.875rem;
  color: #4a5568;
  margin-bottom: 0.75rem;
  font-weight: 500;
}

.suggestions-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.suggestion-btn {
  background: white;
  border: 1px solid #cbd5e0;
  padding: 0.5rem 0.75rem;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.875rem;
}

.suggestion-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.input-container {
  margin-bottom: 0.75rem;
}

.input-wrapper {
  display: flex;
  align-items: flex-end;
  background: #f7fafc;
  border: 2px solid #e2e8f0;
  border-radius: 20px;
  padding: 0.75rem 1rem;
  transition: border-color 0.2s;
}

.input-wrapper:focus-within {
  border-color: #667eea;
}

.message-input {
  flex: 1;
  border: none;
  background: transparent;
  outline: none;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
  line-height: 1.5;
  max-height: 120px;
  min-height: 24px;
}

.message-input::placeholder {
  color: #a0aec0;
}

.input-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-left: 0.75rem;
}

.attachment-btn,
.voice-btn,
.send-btn {
  background: none;
  border: none;
  padding: 0.5rem;
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 1.25rem;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.attachment-btn:hover,
.voice-btn:hover {
  background: #e2e8f0;
}

.voice-btn.active {
  background: #f56565;
  color: white;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.send-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.send-btn:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.sending-icon {
  animation: spin 1s linear infinite;
}

.input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
  color: #718096;
}

.ai-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-indicator.online { background: #48bb78; }
.status-indicator.offline { background: #f56565; }
.status-indicator.thinking { 
  background: #ed8936; 
  animation: pulse 1s infinite;
}

/* Адаптивность */
@media (max-width: 768px) {
  .ai-chat-container {
    width: 100vw;
    height: 100vh;
    max-height: none;
    border-radius: 0;
  }
  
  .chat-header {
    padding: 1rem;
  }
  
  .chat-title {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .chat-icon {
    width: 40px;
    height: 40px;
    font-size: 1.5rem;
  }
  
  .message-content {
    max-width: 85%;
  }
  
  .quick-question-buttons {
    grid-template-columns: 1fr;
  }
  
  .suggestions-list {
    flex-direction: column;
  }
  
  .input-wrapper {
    padding: 0.5rem 0.75rem;
  }
}

@media (max-width: 480px) {
  .chat-header {
    padding: 0.75rem;
  }
  
  .chat-info h3 {
    font-size: 1.125rem;
  }
  
  .message-avatar {
    width: 32px;
    height: 32px;
    margin: 0 0.75rem;
    font-size: 1rem;
  }
  
  .message-text {
    padding: 0.75rem;
  }
  
  .input-actions {
    gap: 0.25rem;
    margin-left: 0.5rem;
  }
  
  .attachment-btn,
  .voice-btn,
  .send-btn {
    width: 32px;
    height: 32px;
    font-size: 1rem;
  }
}
</style>