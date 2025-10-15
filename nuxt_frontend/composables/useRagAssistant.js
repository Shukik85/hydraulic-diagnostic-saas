import { ref } from 'vue'

export const useRagAssistant = () => {
  const messages = ref([])
  const isLoading = ref(false)

  const sendQuery = async (query) => {
    if (!query.trim()) return

    // Добавляем сообщение пользователя
    messages.value.push({
      role: 'user',
      content: query
    })

    isLoading.value = true

    try {
      // TODO: Заменить на реальный API-запрос
      // const response = await $fetch('/api/rag/query', {
      //   method: 'POST',
      //   body: { query }
      // })

      // Тестовая заглушка
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const mockResponse = generateMockResponse(query)
      
      messages.value.push({
        role: 'assistant',
        content: mockResponse
      })
    } catch (error) {
      console.error('Ошибка при отправке запроса:', error)
      messages.value.push({
        role: 'assistant',
        content: 'Извините, произошла ошибка при обработке вашего запроса.'
      })
    } finally {
      isLoading.value = false
    }
  }

  const generateMockResponse = (query) => {
    const responses = [
      `Я получил ваш запрос: "${query}". Это тестовый ответ от RAG-ассистента.`,
      `Спасибо за вопрос! В данный момент работает заглушка. Ваш запрос: "${query}".`,
      `RAG-ассистент (тестовый режим): обрабатываю запрос "${query}". В будущем здесь будет реальный ответ на основе документации.`
    ]
    return responses[Math.floor(Math.random() * responses.length)]
  }

  return {
    messages,
    isLoading,
    sendQuery
  }
}
