/**
 * Сервис для работы с RAG (Retrieval-Augmented Generation) системой
 */
import { api } from './api'

class RAGService {
  constructor() {
    this.baseUrl = '/api/rag-system'
    this.cache = new Map()
    this.cacheTimeout = 5 * 60 * 1000 // 5 минут
  }

  /**
   * Поиск в базе знаний
   */
  async searchKnowledge(query, topK = 5) {
    try {
      const cacheKey = `search_${query}_${topK}`
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      const response = await api.post(`${this.baseUrl}/search_knowledge/`, {
        query: query.trim(),
        top_k: topK
      })

      this.setCache(cacheKey, response.data)
      return response.data
    } catch (error) {
      console.error('Ошибка поиска знаний:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Получение рекомендаций по симптомам
   */
  async getRecommendations(symptoms, systemType = null) {
    try {
      const cacheKey = `recommendations_${symptoms.join(',')}_${systemType}`
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      const response = await api.post(`${this.baseUrl}/get_recommendations/`, {
        symptoms: Array.isArray(symptoms) ? symptoms : [symptoms],
        system_type: systemType
      })

      this.setCache(cacheKey, response.data)
      return response.data
    } catch (error) {
      console.error('Ошибка получения рекомендаций:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Задать вопрос RAG системе
   */
  async askQuestion(question, context = {}) {
    try {
      if (!question.trim()) {
        throw new Error('Вопрос не может быть пустым')
      }

      const response = await api.post(`${this.baseUrl}/ask_question/`, {
        question: question.trim(),
        context: context
      })

      return response.data
    } catch (error) {
      console.error('Ошибка обработки вопроса:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Получение статистики базы знаний
   */
  async getKnowledgeStats() {
    try {
      const cacheKey = 'knowledge_stats'
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      const response = await api.get(`${this.baseUrl}/knowledge_stats/`)
      
      this.setCache(cacheKey, response.data, 60000) // Кеш на 1 минуту
      return response.data
    } catch (error) {
      console.error('Ошибка получения статистики:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Контекстный поиск для автодополнения
   */
  async getContextualSuggestions(partialQuery, limit = 5) {
    try {
      if (partialQuery.length < 3) {
        return { suggestions: [] }
      }

      const cacheKey = `suggestions_${partialQuery}_${limit}`
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      // Поиск с частичным запросом
      const searchResults = await this.searchKnowledge(partialQuery, limit)
      
      // Генерация предложений на основе найденных документов
      const suggestions = this.generateSuggestions(searchResults.results, partialQuery)
      
      const result = { suggestions }
      this.setCache(cacheKey, result, 30000) // Кеш на 30 секунд
      
      return result
    } catch (error) {
      console.error('Ошибка получения предложений:', error)
      return { suggestions: [] }
    }
  }

  /**
   * Генерация предложений на основе результатов поиска
   */
  generateSuggestions(searchResults, partialQuery) {
    const suggestions = new Set()
    const query = partialQuery.toLowerCase()
    
    // Предопределенные шаблоны вопросов
    const questionTemplates = [
      'Что означает {term}?',
      'Как диагностировать {term}?',
      'Причины {term}',
      'Как устранить {term}?',
      'Нормальные значения {term}',
      'ГОСТ стандарты для {term}',
      'Профилактика {term}',
      'Симптомы {term}'
    ]
    
    // Извлечение ключевых терминов из результатов поиска
    const keyTerms = new Set()
    
    searchResults.forEach(result => {
      const title = result.title.toLowerCase()
      const content = result.content.toLowerCase()
      
      // Поиск технических терминов
      const technicalTerms = [
        'давление', 'температура', 'вибрация', 'расход',
        'насос', 'фильтр', 'масло', 'клапан', 'уплотнение',
        'утечка', 'кавитация', 'перегрев', 'износ'
      ]
      
      technicalTerms.forEach(term => {
        if (title.includes(term) || content.includes(term)) {
          keyTerms.add(term)
        }
      })
    })
    
    // Генерация предложений на основе терминов
    keyTerms.forEach(term => {
      if (term.includes(query) || query.includes(term)) {
        questionTemplates.forEach(template => {
          const suggestion = template.replace('{term}', term)
          suggestions.add(suggestion)
        })
      }
    })
    
    // Добавление частых вопросов
    const commonQuestions = [
      'Нормальное давление в гидросистеме',
      'Признаки износа гидронасоса',
      'Как часто менять гидравлическое масло',
      'Причины перегрева гидросистемы',
      'Диагностика утечек в гидроприводе',
      'ГОСТ требования к гидравлическому оборудованию'
    ]
    
    commonQuestions.forEach(question => {
      if (question.toLowerCase().includes(query)) {
        suggestions.add(question)
      }
    })
    
    return Array.from(suggestions).slice(0, 5)
  }

  /**
   * Семантический поиск похожих проблем
   */
  async findSimilarProblems(problemDescription, systemType = null) {
    try {
      const response = await api.post(`${this.baseUrl}/find_similar_problems/`, {
        problem_description: problemDescription,
        system_type: systemType,
        top_k: 10
      })

      return response.data
    } catch (error) {
      console.error('Ошибка поиска похожих проблем:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Анализ данных датчиков с помощью RAG
   */
  async analyzeSensorData(sensorData, systemInfo = {}) {
    try {
      const response = await api.post(`${this.baseUrl}/analyze_sensor_data/`, {
        sensor_data: sensorData,
        system_info: systemInfo
      })

      return response.data
    } catch (error) {
      console.error('Ошибка анализа данных датчиков:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Получение экспертных рекомендаций по ГОСТ
   */
  async getGOSTRecommendations(systemType, parameters = {}) {
    try {
      const cacheKey = `gost_${systemType}_${JSON.stringify(parameters)}`
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      const response = await api.post(`${this.baseUrl}/gost_recommendations/`, {
        system_type: systemType,
        parameters: parameters
      })

      this.setCache(cacheKey, response.data)
      return response.data
    } catch (error) {
      console.error('Ошибка получения ГОСТ рекомендаций:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Добавление нового знания в базу
   */
  async addKnowledge(document) {
    try {
      const response = await api.post(`${this.baseUrl}/add_knowledge/`, {
        title: document.title,
        content: document.content,
        category: document.category || 'user_generated',
        tags: document.tags || [],
        source: document.source || 'user_input'
      })

      // Очистка кеша после добавления нового знания
      this.clearCache()
      
      return response.data
    } catch (error) {
      console.error('Ошибка добавления знания:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Обратная связь по качеству ответа
   */
  async provideFeedback(questionId, rating, comment = '') {
    try {
      const response = await api.post(`${this.baseUrl}/feedback/`, {
        question_id: questionId,
        rating: rating, // 1-5 stars
        comment: comment
      })

      return response.data
    } catch (error) {
      console.error('Ошибка отправки обратной связи:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Получение часто задаваемых вопросов
   */
  async getFAQ(category = null, limit = 10) {
    try {
      const cacheKey = `faq_${category}_${limit}`
      const cached = this.getFromCache(cacheKey)
      
      if (cached) {
        return cached
      }

      const params = { limit }
      if (category) {
        params.category = category
      }

      const response = await api.get(`${this.baseUrl}/faq/`, { params })
      
      this.setCache(cacheKey, response.data, 120000) // Кеш на 2 минуты
      return response.data
    } catch (error) {
      console.error('Ошибка получения FAQ:', error)
      throw this.handleError(error)
    }
  }

  /**
   * Интеллектуальная категоризация вопроса
   */
  classifyQuestion(question) {
    const categories = {
      'diagnostic': ['диагностика', 'проблема', 'неисправность', 'ошибка', 'сбой'],
      'maintenance': ['обслуживание', 'замена', 'ремонт', 'профилактика'],
      'standards': ['гост', 'стандарт', 'норма', 'требование'],
      'technical': ['принцип', 'устройство', 'схема', 'конструкция'],
      'parameters': ['давление', 'температура', 'расход', 'скорость', 'мощность']
    }
    
    const questionLower = question.toLowerCase()
    
    for (const [category, keywords] of Object.entries(categories)) {
      if (keywords.some(keyword => questionLower.includes(keyword))) {
        return category
      }
    }
    
    return 'general'
  }

  /**
   * Извлечение ключевых сущностей из вопроса
   */
  extractEntities(question) {
    const entities = {
      systems: [],
      components: [],
      parameters: [],
      actions: []
    }
    
    const questionLower = question.toLowerCase()
    
    // Системы
    const systems = ['гидропривод', 'гидросистема', 'насос', 'компрессор', 'привод']
    systems.forEach(system => {
      if (questionLower.includes(system)) {
        entities.systems.push(system)
      }
    })
    
    // Компоненты
    const components = ['фильтр', 'клапан', 'уплотнение', 'подшипник', 'масло']
    components.forEach(component => {
      if (questionLower.includes(component)) {
        entities.components.push(component)
      }
    })
    
    // Параметры
    const parameters = ['давление', 'температура', 'вибрация', 'расход', 'мощность']
    parameters.forEach(parameter => {
      if (questionLower.includes(parameter)) {
        entities.parameters.push(parameter)
      }
    })
    
    // Действия
    const actions = ['проверить', 'заменить', 'настроить', 'отрегулировать', 'диагностировать']
    actions.forEach(action => {
      if (questionLower.includes(action)) {
        entities.actions.push(action)
      }
    })
    
    return entities
  }

  /**
   * Кеширование
   */
  getFromCache(key) {
    const cached = this.cache.get(key)
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data
    }
    this.cache.delete(key)
    return null
  }

  setCache(key, data, timeout = null) {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      timeout: timeout || this.cacheTimeout
    })
  }

  clearCache() {
    this.cache.clear()
  }

  /**
   * Обработка ошибок
   */
  handleError(error) {
    if (error.response) {
      // Сервер вернул ошибку
      const message = error.response.data?.message || 'Ошибка сервера'
      return new Error(message)
    } else if (error.request) {
      // Запрос был отправлен, но ответа нет
      return new Error('Сервер недоступен')
    } else {
      // Ошибка настройки запроса
      return new Error(error.message || 'Неизвестная ошибка')
    }
  }
}

// Экспорт экземпляра сервиса
export const ragService = new RAGService()

// Vue композабл для работы с RAG
export function useRAG() {
  const isLoading = ref(false)
  const error = ref(null)
  const searchResults = ref([])
  const suggestions = ref([])
  
  const searchKnowledge = async (query, topK = 5) => {
    isLoading.value = true
    error.value = null
    
    try {
      const result = await ragService.searchKnowledge(query, topK)
      searchResults.value = result.results || []
      return result
    } catch (err) {
      error.value = err.message
      searchResults.value = []
      throw err
    } finally {
      isLoading.value = false
    }
  }
  
  const askQuestion = async (question, context = {}) => {
    isLoading.value = true
    error.value = null
    
    try {
      const result = await ragService.askQuestion(question, context)
      return result
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }
  
  const getRecommendations = async (symptoms, systemType = null) => {
    isLoading.value = true
    error.value = null
    
    try {
      const result = await ragService.getRecommendations(symptoms, systemType)
      return result
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }
  
  const getSuggestions = async (partialQuery) => {
    if (partialQuery.length < 3) {
      suggestions.value = []
      return
    }
    
    try {
      const result = await ragService.getContextualSuggestions(partialQuery)
      suggestions.value = result.suggestions || []
    } catch (err) {
      suggestions.value = []
    }
  }
  
  const clearResults = () => {
    searchResults.value = []
    suggestions.value = []
    error.value = null
  }
  
  return {
    isLoading: readonly(isLoading),
    error: readonly(error),
    searchResults: readonly(searchResults),
    suggestions: readonly(suggestions),
    searchKnowledge,
    askQuestion,
    getRecommendations,
    getSuggestions,
    clearResults
  }
}

export default ragService