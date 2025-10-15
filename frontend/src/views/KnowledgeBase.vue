<template>
  <div class="knowledge-base-container">
    <!-- Заголовок -->
    <div class="kb-header">
      <div class="header-content">
        <h1 class="kb-title">
          📚 Base знаний ГОСТ
        </h1>
        <p class="kb-subtitle">
          Поиск по технической документации и стандартам гидравлического оборудования
        </p>
      </div>
      
      <div class="header-stats">
        <div class="stat-item">
          <div class="stat-value">{{ knowledgeStats.total_documents || 0 }}</div>
          <div class="stat-label">Документов</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ knowledgeStats.total_categories || 0 }}</div>
          <div class="stat-label">Категорий</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ knowledgeStats.last_updated_days || 0 }}</div>
          <div class="stat-label">Дней назад</div>
        </div>
      </div>
    </div>

    <!-- Поиск и фильтры -->
    <div class="search-section">
      <div class="search-container">
        <!-- Основной поиск -->
        <div class="main-search">
          <div class="search-input-wrapper">
            <input
              v-model="searchQuery"
              @input="handleSearchInput"
              @keydown.enter="performSearch"
              placeholder="Поиск по базе знаний ГОСТ..."
              class="search-input"
              ref="searchInput"
            >
            
            <div class="search-actions">
              <button 
                class="search-btn" 
                @click="performSearch"
                :disabled="isSearching || !searchQuery.trim()"
              >
                <span v-if="isSearching" class="search-spinner">🔄</span>
                <span v-else>🔍</span>
              </button>
              
              <button 
                class="voice-search-btn" 
                @click="startVoiceSearch"
                :class="{ active: isListening }"
                title="Голосовой поиск"
              >
                🎤
              </button>
            </div>
          </div>
          
          <!-- Автодополнение -->
          <div v-if="searchSuggestions.length > 0" class="search-suggestions">
            <div 
              v-for="suggestion in searchSuggestions" 
              :key="suggestion"
              class="suggestion-item"
              @click="selectSuggestion(suggestion)"
            >
              {{ suggestion }}
            </div>
          </div>
        </div>

        <!-- Быстрый поиск по категориям -->
        <div class="quick-categories">
          <h3>Быстрый поиск:</h3>
          <div class="category-buttons">
            <button 
              v-for="category in quickCategories" 
              :key="category.id"
              class="category-btn"
              :class="{ active: selectedCategories.includes(category.id) }"
              @click="toggleCategory(category.id)"
            >
              {{ category.icon }} {{ category.name }}
            </button>
          </div>
        </div>
      </div>

      <!-- Расширенные фильтры -->
      <div class="filters-section" :class="{ expanded: showAdvancedFilters }">
        <div class="filters-header">
          <button 
            class="toggle-filters-btn" 
            @click="showAdvancedFilters = !showAdvancedFilters"
          >
            {{ showAdvancedFilters ? '➖' : '➕' }} Расширенные фильтры
          </button>
        </div>
        
        <div v-if="showAdvancedFilters" class="filters-content">
          <div class="filter-group">
            <label>Тип документа:</label>
            <div class="filter-options">
              <label v-for="type in documentTypes" :key="type.id" class="filter-option">
                <input 
                  type="checkbox" 
                  :value="type.id" 
                  v-model="selectedDocTypes"
                  @change="applyFilters"
                >
                <span>{{ type.name }}</span>
              </label>
            </div>
          </div>
          
          <div class="filter-group">
            <label>Дата обновления:</label>
            <select v-model="dateFilter" @change="applyFilters" class="filter-select">
              <option value="">Любая</option>
              <option value="week">За неделю</option>
              <option value="month">За месяц</option>
              <option value="quarter">За квартал</option>
              <option value="year">За год</option>
            </select>
          </div>
          
          <div class="filter-group">
            <label>Релевантность:</label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              v-model="relevanceThreshold"
              @input="applyFilters"
              class="relevance-slider"
            >
            <span class="relevance-value">{{ relevanceThreshold }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Результаты поиска -->
    <div class="search-results">
      <!-- Статистика поиска -->
      <div v-if="searchResults.length > 0" class="results-stats">
        <div class="results-info">
          Найдено {{ searchResults.length }} результатов за {{ searchTime }}ms
        </div>
        
        <div class="results-actions">
          <button class="action-btn" @click="exportResults">
            📤 Экспорт
          </button>
          <button class="action-btn" @click="generateReport">
            📋 Создать отчет
          </button>
        </div>
      </div>

      <!-- Список результатов -->
      <div v-if="searchResults.length > 0" class="results-list">
        <div 
          v-for="(result, index) in paginatedResults" 
          :key="result.id || index"
          class="result-item"
          :class="{ expanded: expandedResults.has(result.id) }"
        >
          <div class="result-header" @click="toggleResult(result.id)">
            <div class="result-info">
              <div class="result-title">{{ result.title }}</div>
              <div class="result-meta">
                <span class="result-category">{{ getCategoryName(result.category) }}</span>
                <span class="result-relevance">
                  Релевантность: {{ Math.round(result.relevance_score * 100) }}%
                </span>
                <span class="result-date">{{ formatDate(result.updated_at) }}</span>
              </div>
            </div>
            
            <div class="result-actions">
              <div class="relevance-bar">
                <div 
                  class="relevance-fill" 
                  :style="{ width: (result.relevance_score * 100) + '%' }"
                ></div>
              </div>
              <button class="expand-btn">
                {{ expandedResults.has(result.id) ? '▲' : '▼' }}
              </button>
            </div>
          </div>
          
          <!-- Краткое описание -->
          <div class="result-snippet">
            {{ result.snippet || result.content?.substring(0, 200) + '...' }}
          </div>
          
          <!-- Развернутое содержимое -->
          <div v-if="expandedResults.has(result.id)" class="result-content">
            <div class="content-text" v-html="highlightSearchTerms(result.content)"></div>
            
            <div class="content-meta">
              <div class="tags" v-if="result.tags && result.tags.length > 0">
                <span class="tag" v-for="tag in result.tags" :key="tag">
                  #{{ tag }}
                </span>
              </div>
              
              <div class="content-actions">
                <button class="content-btn" @click="copyToClipboard(result.content)">
                  📋 Копировать
                </button>
                <button class="content-btn" @click="askAboutDocument(result)">
                  💬 Спросить AI
                </button>
                <button class="content-btn" @click="addToFavorites(result)">
                  ⭐ В избранное
                </button>
                <a 
                  v-if="result.source_url" 
                  :href="result.source_url" 
                  target="_blank" 
                  class="content-btn link-btn"
                >
                  🔗 Источник
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Пагинация -->
      <div v-if="totalPages > 1" class="pagination">
        <button 
          class="page-btn" 
          @click="goToPage(1)"
          :disabled="currentPage === 1"
        >
          ⏮️
        </button>
        
        <button 
          class="page-btn" 
          @click="goToPage(currentPage - 1)"
          :disabled="currentPage === 1"
        >
          ⬅️
        </button>
        
        <div class="page-numbers">
          <button 
            v-for="page in visiblePages" 
            :key="page"
            class="page-btn"
            :class="{ active: page === currentPage }"
            @click="goToPage(page)"
          >
            {{ page }}
          </button>
        </div>
        
        <button 
          class="page-btn" 
          @click="goToPage(currentPage + 1)"
          :disabled="currentPage === totalPages"
        >
          ➡️
        </button>
        
        <button 
          class="page-btn" 
          @click="goToPage(totalPages)"
          :disabled="currentPage === totalPages"
        >
          ⏭️
        </button>
      </div>

      <!-- Пустое состояние -->
      <div v-if="hasSearched && searchResults.length === 0" class="no-results">
        <div class="no-results-icon">🔍</div>
        <h3>Результаты не найдены</h3>
        <p>Попробуйте изменить запрос или использовать другие ключевые слова.</p>
        
        <div class="search-suggestions-empty">
          <h4>Возможно, вас интересует:</h4>
          <div class="suggestion-buttons">
            <button 
              v-for="suggestion in popularQueries" 
              :key="suggestion"
              class="suggestion-btn"
              @click="searchQuery = suggestion; performSearch()"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>
      </div>

      <!-- Популярные документы -->
      <div v-if="!hasSearched" class="popular-documents">
        <h2>📊 Популярные документы</h2>
        
        <div class="popular-categories">
          <div 
            v-for="category in popularCategories" 
            :key="category.id"
            class="popular-category"
          >
            <h3>{{ category.icon }} {{ category.name }}</h3>
            <div class="category-documents">
              <div 
                v-for="doc in category.documents" 
                :key="doc.id"
                class="popular-doc"
                @click="viewDocument(doc)"
              >
                <div class="doc-title">{{ doc.title }}</div>
                <div class="doc-views">{{ doc.views || 0 }} просмотров</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- FAQ секция -->
    <div v-if="!hasSearched" class="faq-section">
      <h2>❓ Часто задаваемые вопросы</h2>
      
      <div class="faq-list">
        <details 
          v-for="faq in frequentlyAsked" 
          :key="faq.id"
          class="faq-item"
        >
          <summary class="faq-question">{{ faq.question }}</summary>
          <div class="faq-answer" v-html="faq.answer"></div>
          <div class="faq-actions">
            <button class="faq-action-btn" @click="searchQuery = faq.related_query; performSearch()">
              🔍 Найти похожее
            </button>
            <button class="faq-action-btn" @click="askAboutFAQ(faq)">
              💬 Спросить AI
            </button>
          </div>
        </details>
      </div>
    </div>

    <!-- Модальные окна -->
    <AIChat v-if="showAIChat" @close="showAIChat = false" />
    
    <DocumentModal 
      v-if="selectedDocument"
      :document="selectedDocument"
      @close="selectedDocument = null"
    />
  </div>
</template>

<script>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useRAG } from '@/services/ragService'
import AIChat from '@/components/AIChat.vue'
import DocumentModal from '@/components/DocumentModal.vue'

export default {
  name: 'KnowledgeBase',
  components: {
    AIChat,
    DocumentModal
  },
  setup() {
    const router = useRouter()
    const route = useRoute()
    const { 
      isLoading, 
      error, 
      searchResults, 
      suggestions, 
      searchKnowledge, 
      askQuestion, 
      getSuggestions 
    } = useRAG()
    
    // Реактивные данные
    const searchQuery = ref('')
    const isSearching = ref(false)
    const isListening = ref(false)
    const searchSuggestions = ref([])
    const hasSearched = ref(false)
    const searchTime = ref(0)
    const knowledgeStats = ref({})
    
    // Фильтры
    const showAdvancedFilters = ref(false)
    const selectedCategories = ref([])
    const selectedDocTypes = ref([])
    const dateFilter = ref('')
    const relevanceThreshold = ref(70)
    
    // Результаты
    const expandedResults = ref(new Set())
    const currentPage = ref(1)
    const itemsPerPage = ref(10)
    
    // Модальные окна
    const showAIChat = ref(false)
    const selectedDocument = ref(null)
    
    // Статические данные
    const quickCategories = [
      { id: 'gost', name: 'ГОСТ стандарты', icon: '📋' },
      { id: 'diagnostics', name: 'Диагностика', icon: '🔍' },
      { id: 'maintenance', name: 'Обслуживание', icon: '🔧' },
      { id: 'safety', name: 'Безопасность', icon: '⚠️' },
      { id: 'components', name: 'Компоненты', icon: '⚙️' }
    ]
    
    const documentTypes = [
      { id: 'standard', name: 'Стандарты' },
      { id: 'manual', name: 'Руководства' },
      { id: 'specification', name: 'Спецификации' },
      { id: 'procedure', name: 'Процедуры' },
      { id: 'reference', name: 'Справочники' }
    ]
    
    const popularQueries = [
      'ГОСТ гидравлические системы',
      'диагностика гидронасоса',
      'температура гидромасла',
      'давление в гидроприводе',
      'техническое обслуживание'
    ]
    
    const popularCategories = ref([
      {
        id: 'gost',
        name: 'ГОСТ стандарты',
        icon: '📋',
        documents: [
          { id: 1, title: 'ГОСТ 17398-72 Насосы гидравлические', views: 1250 },
          { id: 2, title: 'ГОСТ 16517-70 Гидроприводы', views: 980 },
          { id: 3, title: 'ГОСТ 6540-68 Масла гидравлические', views: 756 }
        ]
      },
      {
        id: 'diagnostics',
        name: 'Диагностика',
        icon: '🔍',
        documents: [
          { id: 4, title: 'Методы диагностики гидросистем', views: 892 },
          { id: 5, title: 'Анализ вибраций в гидроприводах', views: 634 },
          { id: 6, title: 'Контроль загрязнения масла', views: 543 }
        ]
      }
    ])
    
    const frequentlyAsked = ref([
      {
        id: 1,
        question: 'Какое нормальное рабочее давление в гидросистеме?',
        answer: 'Рабочее давление зависит от типа системы. Для промышленных гидроприводов обычно составляет 160-320 бар, для мобильной техники - 200-350 бар.',
        related_query: 'рабочее давление гидросистема ГОСТ'
      },
      {
        id: 2,
        question: 'Как часто нужно менять гидравлическое масло?',
        answer: 'Согласно ГОСТ, замена масла производится каждые 2000-4000 моточасов или раз в год, в зависимости от условий эксплуатации.',
        related_query: 'замена гидравлического масла периодичность'
      }
    ])
    
    // Вычисляемые свойства
    const paginatedResults = computed(() => {
      const start = (currentPage.value - 1) * itemsPerPage.value
      const end = start + itemsPerPage.value
      return searchResults.value.slice(start, end)
    })
    
    const totalPages = computed(() => {
      return Math.ceil(searchResults.value.length / itemsPerPage.value)
    })
    
    const visiblePages = computed(() => {
      const total = totalPages.value
      const current = currentPage.value
      const delta = 2
      
      let start = Math.max(1, current - delta)
      let end = Math.min(total, current + delta)
      
      if (end - start < 2 * delta) {
        start = Math.max(1, end - 2 * delta)
        end = Math.min(total, start + 2 * delta)
      }
      
      const pages = []
      for (let i = start; i <= end; i++) {
        pages.push(i)
      }
      
      return pages
    })
    
    // Методы
    const loadKnowledgeStats = async () => {
      try {
        // Имитация загрузки статистики
        knowledgeStats.value = {
          total_documents: 1250,
          total_categories: 12,
          last_updated_days: 3
        }
      } catch (error) {
        console.error('Ошибка загрузки статистики:', error)
      }
    }
    
    const handleSearchInput = async () => {
      const query = searchQuery.value.trim()
      
      if (query.length >= 3) {
        try {
          await getSuggestions(query)
          searchSuggestions.value = suggestions.value
        } catch (error) {
          searchSuggestions.value = []
        }
      } else {
        searchSuggestions.value = []
      }
    }
    
    const performSearch = async () => {
      const query = searchQuery.value.trim()
      if (!query) return
      
      isSearching.value = true
      hasSearched.value = true
      searchSuggestions.value = []
      
      const startTime = Date.now()
      
      try {
        await searchKnowledge(query, 20)
        searchTime.value = Date.now() - startTime
        currentPage.value = 1
        
        // Обновление URL для обратной навигации
        router.push({
          query: { 
            ...route.query, 
            q: query,
            categories: selectedCategories.value.join(','),
            types: selectedDocTypes.value.join(',')
          }
        })
        
      } catch (error) {
        console.error('Ошибка поиска:', error)
      } finally {
        isSearching.value = false
      }
    }
    
    const selectSuggestion = (suggestion) => {
      searchQuery.value = suggestion
      searchSuggestions.value = []
      performSearch()
    }
    
    const startVoiceSearch = () => {
      if (!('webkitSpeechRecognition' in window)) {
        alert('Голосовой поиск не поддерживается в этом браузере')
        return
      }
      
      const recognition = new webkitSpeechRecognition()
      recognition.lang = 'ru-RU'
      recognition.continuous = false
      recognition.interimResults = false
      
      recognition.onstart = () => {
        isListening.value = true
      }
      
      recognition.onresult = (event) => {
        const result = event.results.transcript
        searchQuery.value = result
        performSearch()
      }
      
      recognition.onerror = (event) => {
        console.error('Ошибка голосового поиска:', event.error)
      }
      
      recognition.onend = () => {
        isListening.value = false
      }
      
      recognition.start()
    }
    
    const toggleCategory = (categoryId) => {
      const index = selectedCategories.value.indexOf(categoryId)
      if (index > -1) {
        selectedCategories.value.splice(index, 1)
      } else {
        selectedCategories.value.push(categoryId)
      }
      applyFilters()
    }
    
    const applyFilters = () => {
      if (hasSearched.value) {
        performSearch()
      }
    }
    
    const toggleResult = (resultId) => {
      if (expandedResults.value.has(resultId)) {
        expandedResults.value.delete(resultId)
      } else {
        expandedResults.value.add(resultId)
      }
    }
    
    const highlightSearchTerms = (content) => {
      if (!searchQuery.value || !content) return content
      
      const query = searchQuery.value.trim()
      const regex = new RegExp(`(${query})`, 'gi')
      return content.replace(regex, '<mark>$1</mark>')
    }
    
    const getCategoryName = (categoryId) => {
      const category = quickCategories.find(c => c.id === categoryId)
      return category ? category.name : categoryId
    }
    
    const formatDate = (dateString) => {
      if (!dateString) return 'Не указано'
      return new Date(dateString).toLocaleDateString('ru-RU')
    }
    
    const copyToClipboard = async (text) => {
      try {
        await navigator.clipboard.writeText(text)
        // Показать уведомление об успешном копировании
      } catch (error) {
        console.error('Ошибка копирования:', error)
      }
    }
    
    const askAboutDocument = (document) => {
      // Открытие AI чата с контекстом документа
      showAIChat.value = true
      // TODO: Передать контекст документа в AI чат
    }
    
    const addToFavorites = (document) => {
      // Добавление в избранное
      console.log('Добавление в избранное:', document.title)
    }
    
    const viewDocument = (document) => {
      selectedDocument.value = document
    }
    
    const askAboutFAQ = (faq) => {
      showAIChat.value = true
      // TODO: Передать контекст FAQ в AI чат
    }
    
    const exportResults = () => {
      const dataToExport = {
        query: searchQuery.value,
        results: searchResults.value,
        filters: {
          categories: selectedCategories.value,
          document_types: selectedDocTypes.value,
          date_filter: dateFilter.value,
          relevance_threshold: relevanceThreshold.value
        },
        exported_at: new Date().toISOString()
      }
      
      const dataStr = JSON.stringify(dataToExport, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      
      const link = document.createElement('a')
      link.href = URL.createObjectURL(dataBlob)
      link.download = `knowledge_base_search_${new Date().toISOString().split('T')}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    
    const generateReport = () => {
      // Генерация отчета на основе результатов поиска
      console.log('Генерация отчета для:', searchResults.value.length, 'документов')
    }
    
    const goToPage = (page) => {
      currentPage.value = page
      // Прокрутка к началу результатов
      document.querySelector('.search-results')?.scrollIntoView({ behavior: 'smooth' })
    }
    
    // Инициализация из URL параметров
    const initializeFromURL = () => {
      const query = route.query.q
      if (query) {
        searchQuery.value = query
        hasSearched.value = true
        performSearch()
      }
      
      if (route.query.categories) {
        selectedCategories.value = route.query.categories.split(',')
      }
      
      if (route.query.types) {
        selectedDocTypes.value = route.query.types.split(',')
      }
    }
    
    // Наблюдатели
    watch(searchQuery, (newQuery) => {
      if (!newQuery) {
        searchSuggestions.value = []
        hasSearched.value = false
      }
    })
    
    // Жизненный цикл
    onMounted(() => {
      loadKnowledgeStats()
      initializeFromURL()
      
      // Фокус на поле поиска
      nextTick(() => {
        const searchInput = document.querySelector('.search-input')
        if (searchInput) {
          searchInput.focus()
        }
      })
    })
    
    return {
      searchQuery,
      isSearching,
      isListening,
      searchSuggestions,
      hasSearched,
      searchTime,
      knowledgeStats,
      showAdvancedFilters,
      selectedCategories,
      selectedDocTypes,
      dateFilter,
      relevanceThreshold,
      searchResults,
      expandedResults,
      currentPage,
      totalPages,
      visiblePages,
      paginatedResults,
      showAIChat,
      selectedDocument,
      quickCategories,
      documentTypes,
      popularQueries,
      popularCategories,
      frequentlyAsked,
      handleSearchInput,
      performSearch,
      selectSuggestion,
      startVoiceSearch,
      toggleCategory,
      applyFilters,
      toggleResult,
      highlightSearchTerms,
      getCategoryName,
      formatDate,
      copyToClipboard,
      askAboutDocument,
      addToFavorites,
      viewDocument,
      askAboutFAQ,
      exportResults,
      generateReport,
      goToPage
    }
  }
}
</script>

<style scoped>
/* Основные стили Knowledge Base */
.knowledge-base-container {
  padding: 0;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  min-height: 100vh;
}

/* Заголовок */
.kb-header {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 3rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 2rem;
}

.kb-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
}

.kb-subtitle {
  font-size: 1.125rem;
  margin: 0;
  opacity: 0.9;
}

.header-stats {
  display: flex;
  gap: 2rem;
}

.stat-item {
  text-align: center;
  background: rgba(255, 255, 255, 0.1);
  padding: 1rem;
  border-radius: 12px;
  min-width: 100px;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  display: block;
  margin-bottom: 0.25rem;
}

.stat-label {
  font-size: 0.875rem;
  opacity: 0.9;
}

/* Секция поиска */
.search-section {
  background: white;
  margin: -2rem 2rem 2rem;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  position: relative;
  z-index: 10;
}

.search-container {
  margin-bottom: 2rem;
}

.main-search {
  position: relative;
  margin-bottom: 2rem;
}

.search-input-wrapper {
  display: flex;
  align-items: center;
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem;
  transition: border-color 0.2s;
}

.search-input-wrapper:focus-within {
  border-color: #667eea;
}

.search-input {
  flex: 1;
  border: none;
  background: transparent;
  outline: none;
  font-size: 1.125rem;
  color: #2d3748;
}

.search-input::placeholder {
  color: #a0aec0;
}

.search-actions {
  display: flex;
  gap: 0.5rem;
  margin-left: 1rem;
}

.search-btn,
.voice-search-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 1.125rem;
  min-width: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.search-btn:hover,
.voice-search-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.search-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.voice-search-btn.active {
  background: #f56565;
  animation: pulse 1s infinite;
}

.search-spinner {
  animation: spin 1s linear infinite;
}

/* Автодополнение */
.search-suggestions {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: white;
  border: 1px solid #e2e8f0;
  border-top: none;
  border-radius: 0 0 8px 8px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  z-index: 100;
  max-height: 300px;
  overflow-y: auto;
}

.suggestion-item {
  padding: 0.75rem 1rem;
  cursor: pointer;
  transition: background 0.2s;
  border-bottom: 1px solid #f1f5f9;
}

.suggestion-item:hover {
  background: #f8fafc;
}

.suggestion-item:last-child {
  border-bottom: none;
}

/* Быстрые категории */
.quick-categories h3 {
  color: #2d3748;
  margin: 0 0 1rem 0;
  font-size: 1.125rem;
}

.category-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.category-btn {
  background: white;
  border: 2px solid #e2e8f0;
  padding: 0.75rem 1rem;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.category-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.category-btn.active {
  border-color: #667eea;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

/* Фильтры */
.filters-section {
  border-top: 1px solid #e2e8f0;
  padding-top: 1.5rem;
}

.toggle-filters-btn {
  background: none;
  border: none;
  color: #667eea;
  font-weight: 600;
  cursor: pointer;
  padding: 0.5rem 0;
  transition: color 0.2s;
}

.toggle-filters-btn:hover {
  color: #553c9a;
}

.filters-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 1.5rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.filter-group label {
  font-weight: 600;
  color: #2d3748;
}

.filter-options {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.filter-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-weight: normal;
}

.filter-select,
.relevance-slider {
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: white;
}

.relevance-slider {
  width: 100%;
}

.relevance-value {
  font-weight: 600;
  color: #667eea;
}

/* Результаты поиска */
.search-results {
  padding: 0 2rem 2rem;
}

.results-stats {
  background: white;
  padding: 1.5rem 2rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.results-info {
  color: #4a5568;
  font-weight: 500;
}

.results-actions {
  display: flex;
  gap: 0.75rem;
}

.action-btn {
  background: linear-gradient(135deg, #4299e1, #3182ce);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(66, 153, 225, 0.3);
}

/* Список результатов */
.results-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.result-item {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  transition: all 0.2s;
}

.result-item:hover {
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.result-header {
  padding: 1.5rem;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #f1f5f9;
}

.result-info {
  flex: 1;
}

.result-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.5rem;
  line-height: 1.4;
}

.result-meta {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  font-size: 0.875rem;
  color: #718096;
}

.result-category {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-weight: 500;
}

.result-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.relevance-bar {
  width: 60px;
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
}

.relevance-fill {
  height: 100%;
  background: linear-gradient(90deg, #f56565, #ed8936, #48bb78);
  transition: width 0.3s;
}

.expand-btn {
  background: none;
  border: none;
  color: #a0aec0;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0.25rem;
  transition: color 0.2s;
}

.expand-btn:hover {
  color: #667eea;
}

.result-snippet {
  padding: 0 1.5rem 1rem;
  color: #4a5568;
  line-height: 1.6;
}

.result-content {
  border-top: 1px solid #f1f5f9;
  padding: 1.5rem;
  background: #f8fafc;
}

.content-text {
  color: #2d3748;
  line-height: 1.7;
  margin-bottom: 1.5rem;
}

.content-text :deep(mark) {
  background: #fef5e7;
  color: #c05621;
  padding: 0.125rem 0.25rem;
  border-radius: 3px;
}

.content-meta {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag {
  background: #e2e8f0;
  color: #4a5568;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.content-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.content-btn {
  background: white;
  border: 1px solid #e2e8f0;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
  color: #4a5568;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.content-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

/* Пагинация */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  margin-top: 2rem;
  padding: 2rem 0;
}

.page-btn {
  background: white;
  border: 1px solid #e2e8f0;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  min-width: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.page-btn:hover:not(:disabled) {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-btn.active {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-color: #667eea;
}

.page-numbers {
  display: flex;
  gap: 0.25rem;
}

/* Пустое состояние */
.no-results {
  text-align: center;
  padding: 4rem 2rem;
  color: #4a5568;
}

.no-results-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  opacity: 0.5;
}

.no-results h3 {
  color: #2d3748;
  margin: 0 0 1rem 0;
}

.search-suggestions-empty {
  margin-top: 2rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.search-suggestions-empty h4 {
  color: #2d3748;
  margin: 0 0 1rem 0;
}

.suggestion-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: center;
}

.suggestion-btn {
  background: white;
  border: 2px solid #e2e8f0;
  padding: 0.75rem 1rem;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.suggestion-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

/* Популярные документы */
.popular-documents {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  margin-bottom: 2rem;
}

.popular-documents h2 {
  color: #2d3748;
  margin: 0 0 2rem 0;
  font-size: 1.5rem;
}

.popular-categories {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.popular-category h3 {
  color: #2d3748;
  margin: 0 0 1rem 0;
  font-size: 1.125rem;
}

.category-documents {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.popular-doc {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  border-left: 3px solid #667eea;
}

.popular-doc:hover {
  background: #f1f5f9;
  transform: translateX(4px);
}

.doc-title {
  font-weight: 500;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.doc-views {
  color: #718096;
  font-size: 0.875rem;
}

/* FAQ */
.faq-section {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.faq-section h2 {
  color: #2d3748;
  margin: 0 0 2rem 0;
  font-size: 1.5rem;
}

.faq-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.faq-item {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}

.faq-question {
  background: #f8fafc;
  padding: 1rem 1.5rem;
  cursor: pointer;
  font-weight: 600;
  color: #2d3748;
  transition: background 0.2s;
}

.faq-question:hover {
  background: #f1f5f9;
}

.faq-answer {
  padding: 1.5rem;
  color: #4a5568;
  line-height: 1.6;
}

.faq-actions {
  padding: 0 1.5rem 1.5rem;
  display: flex;
  gap: 0.75rem;
}

.faq-action-btn {
  background: none;
  border: 1px solid #e2e8f0;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  color: #4a5568;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.faq-action-btn:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

/* Адаптивность */
@media (max-width: 1200px) {
  .filters-content {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  }
}

@media (max-width: 768px) {
  .kb-header {
    flex-direction: column;
    text-align: center;
    padding: 2rem 1rem;
  }
  
  .header-stats {
    justify-content: center;
    flex-wrap: wrap;
  }
  
  .search-section {
    margin: -1rem 1rem 1rem;
    padding: 1.5rem;
  }
  
  .search-results {
    padding: 0 1rem 1rem;
  }
  
  .result-header {
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .result-meta {
    justify-content: center;
  }
  
  .content-actions {
    justify-content: center;
  }
  
  .popular-categories {
    grid-template-columns: 1fr;
  }
  
  .category-buttons {
    justify-content: center;
  }
  
  .page-numbers {
    max-width: 200px;
    overflow-x: auto;
  }
}

@media (max-width: 480px) {
  .kb-title {
    font-size: 2rem;
  }
  
  .search-input {
    font-size: 1rem;
  }
  
  .search-actions {
    margin-left: 0.5rem;
  }
  
  .result-item {
    margin: 0 -0.5rem;
    border-radius: 8px;
  }
  
  .result-header,
  .result-snippet,
  .result-content {
    padding: 1rem;
  }
  
  .filters-content {
    grid-template-columns: 1fr;
  }
  
  .suggestion-buttons,
  .category-buttons {
    flex-direction: column;
    align-items: center;
  }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
