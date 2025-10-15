<template>
  <div class="dashboard-container">
    <!-- Заголовок дашборда -->
    <div class="dashboard-header">
      <div class="header-content">
        <h1 class="dashboard-title">
          🏭 Диагностический дашборд
        </h1>
        <p class="dashboard-subtitle">
          Мониторинг гидравлических систем в реальном времени
        </p>
      </div>
      
      <div class="header-actions">
        <button class="refresh-btn" @click="refreshDashboard" :disabled="isRefreshing">
          <span class="refresh-icon" :class="{ spinning: isRefreshing }">🔄</span>
          {{ isRefreshing ? 'Обновление...' : 'Обновить' }}
        </button>
        
        <div class="auto-refresh-toggle">
          <label class="toggle-switch">
            <input type="checkbox" v-model="autoRefresh" @change="toggleAutoRefresh">
            <span class="slider"></span>
          </label>
          <span class="toggle-label">Авто-обновление</span>
        </div>
      </div>
    </div>

    <!-- Метрики верхнего уровня -->
    <div class="metrics-grid">
      <div class="metric-card total-systems">
        <div class="metric-icon">🏭</div>
        <div class="metric-content">
          <div class="metric-value">{{ dashboardStats.user_systems?.total || 0 }}</div>
          <div class="metric-label">Всего систем</div>
          <div class="metric-change positive">
            +{{ dashboardStats.user_systems?.active || 0 }} активных
          </div>
        </div>
      </div>

      <div class="metric-card active-monitoring">
        <div class="metric-icon">📊</div>
        <div class="metric-content">
          <div class="metric-value">{{ dashboardStats.recent_activity?.sensor_data_points || 0 }}</div>
          <div class="metric-label">Точек данных за 24ч</div>
          <div class="metric-change" :class="{ 'negative': dashboardStats.recent_activity?.critical_events > 0 }">
            {{ dashboardStats.recent_activity?.critical_events || 0 }} критических
          </div>
        </div>
      </div>

      <div class="metric-card diagnostics-ran">
        <div class="metric-icon">🔍</div>
        <div class="metric-content">
          <div class="metric-value">{{ dashboardStats.recent_activity?.diagnostic_reports || 0 }}</div>
          <div class="metric-label">Отчетов за неделю</div>
          <div class="metric-change positive">
            {{ Math.round((dashboardStats.recent_activity?.diagnostic_reports || 0) / 7) }} в день
          </div>
        </div>
      </div>

      <div class="metric-card system-health">
        <div class="metric-icon">❤️</div>
        <div class="metric-content">
          <div class="metric-value">{{ overallHealthScore }}%</div>
          <div class="metric-label">Общее здоровье</div>
          <div class="metric-change" :class="healthChangeClass">
            {{ healthStatus }}
          </div>
        </div>
      </div>
    </div>

    <!-- Основной контент дашборда -->
    <div class="dashboard-main">
      
      <!-- Левая колонка -->
      <div class="dashboard-left">
        
        <!-- Системы требующие внимания -->
        <div class="dashboard-section">
          <div class="section-header">
            <h2 class="section-title">
              ⚠️ Требуют внимания
            </h2>
            <span class="section-count">{{ attentionSystems.length }}</span>
          </div>
          
          <div class="attention-systems">
            <div v-if="attentionSystems.length === 0" class="no-issues">
              <div class="no-issues-icon">✅</div>
              <p>Все системы работают нормально!</p>
            </div>
            
            <div 
              v-for="system in attentionSystems" 
              :key="system.id"
              class="attention-system-card"
              @click="navigateToSystem(system.id)"
            >
              <div class="system-header">
                <div class="system-name">{{ system.name }}</div>
                <div class="critical-badge">{{ system.critical_count }} проблем</div>
              </div>
              
              <div class="system-issues">
                <div class="issue-item" v-for="issue in system.recent_issues" :key="issue.id">
                  <span class="issue-type">{{ getIssueIcon(issue.type) }}</span>
                  <span class="issue-text">{{ issue.message }}</span>
                  <span class="issue-time">{{ formatTimeAgo(issue.timestamp) }}</span>
                </div>
              </div>
              
              <div class="system-actions">
                <button class="action-btn diagnose" @click.stop="runDiagnosis(system.id)">
                  🔍 Диагностика
                </button>
                <button class="action-btn view" @click.stop="viewSystemDetails(system.id)">
                  👁️ Подробнее
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Последние отчеты -->
        <div class="dashboard-section">
          <div class="section-header">
            <h2 class="section-title">
              📋 Последние отчеты
            </h2>
            <router-link to="/reports" class="section-link">Все отчеты →</router-link>
          </div>
          
          <div class="recent-reports">
            <div 
              v-for="report in recentReports" 
              :key="report.id"
              class="report-card"
              :class="`report-${report.severity}`"
            >
              <div class="report-header">
                <div class="report-severity">
                  {{ getSeverityIcon(report.severity) }}
                </div>
                <div class="report-info">
                  <div class="report-title">{{ report.title }}</div>
                  <div class="report-system">{{ report.system_name }}</div>
                </div>
                <div class="report-time">
                  {{ formatDate(report.created_at) }}
                </div>
              </div>
              
              <div class="report-description">
                {{ truncateText(report.description, 100) }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Правая колонка -->
      <div class="dashboard-right">
        
        <!-- Состояние систем (круговая диаграмма) -->
        <div class="dashboard-section">
          <div class="section-header">
            <h2 class="section-title">
              ⚙️ Состояние систем
            </h2>
          </div>
          
          <div class="systems-status-chart">
            <canvas ref="systemsChart" width="300" height="300"></canvas>
            
            <div class="chart-legend">
              <div class="legend-item active">
                <div class="legend-color active"></div>
                <span>Активные ({{ dashboardStats.user_systems?.active || 0 }})</span>
              </div>
              <div class="legend-item maintenance">
                <div class="legend-color maintenance"></div>
                <span>Обслуживание ({{ dashboardStats.user_systems?.maintenance || 0 }})</span>
              </div>
              <div class="legend-item inactive">
                <div class="legend-color inactive"></div>
                <span>Неактивные ({{ dashboardStats.user_systems?.inactive || 0 }})</span>
              </div>
            </div>
          </div>
        </div>

        <!-- AI Рекомендации -->
        <div class="dashboard-section">
          <div class="section-header">
            <h2 class="section-title">
              🤖 AI Рекомендации
            </h2>
            <button class="ai-chat-btn" @click="openAIChat">
              💬 Чат с AI
            </button>
          </div>
          
          <div class="ai-recommendations">
            <div v-if="aiRecommendations.length === 0" class="no-recommendations">
              <div class="ai-thinking">🤔</div>
              <p>AI анализирует ваши системы...</p>
              <button class="generate-recommendations-btn" @click="generateRecommendations">
                Генерировать рекомендации
              </button>
            </div>
            
            <div 
              v-for="recommendation in aiRecommendations" 
              :key="recommendation.id"
              class="recommendation-card"
              :class="`priority-${recommendation.priority}`"
            >
              <div class="recommendation-header">
                <div class="priority-badge">{{ getPriorityIcon(recommendation.priority) }}</div>
                <div class="recommendation-title">{{ recommendation.title }}</div>
              </div>
              
              <div class="recommendation-description">
                {{ recommendation.description }}
              </div>
              
              <div class="recommendation-actions">
                <button class="recommendation-action" @click="applyRecommendation(recommendation)">
                  ✅ Применить
                </button>
                <button class="recommendation-dismiss" @click="dismissRecommendation(recommendation.id)">
                  ❌ Отклонить
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Быстрая диагностика -->
        <div class="dashboard-section">
          <div class="section-header">
            <h2 class="section-title">
              ⚡ Быстрая диагностика
            </h2>
          </div>
          
          <div class="quick-diagnostics">
            <div class="diagnostic-selector">
              <select v-model="selectedSystemForDiagnosis" class="system-select">
                <option value="">Выберите систему</option>
                <option 
                  v-for="system in availableSystems" 
                  :key="system.id" 
                  :value="system.id"
                >
                  {{ system.name }}
                </option>
              </select>
              
              <button 
                class="diagnose-btn" 
                @click="runQuickDiagnosis"
                :disabled="!selectedSystemForDiagnosis || isDiagnosing"
              >
                <span v-if="isDiagnosing">🔄 Анализ...</span>
                <span v-else>🔍 Диагностировать</span>
              </button>
            </div>
            
            <div v-if="quickDiagnosisResult" class="diagnosis-result">
              <div class="result-header">
                <div class="result-score">
                  Здоровье: {{ quickDiagnosisResult.health_score || 0 }}%
                </div>
                <div class="result-status" :class="`status-${quickDiagnosisResult.status}`">
                  {{ getStatusText(quickDiagnosisResult.status) }}
                </div>
              </div>
              
              <div class="result-issues" v-if="quickDiagnosisResult.issues?.length > 0">
                <div class="issues-title">Обнаруженные проблемы:</div>
                <ul class="issues-list">
                  <li v-for="issue in quickDiagnosisResult.issues" :key="issue">
                    {{ issue }}
                  </li>
                </ul>
              </div>
              
              <button class="full-report-btn" @click="openFullDiagnosisReport">
                📊 Полный отчет
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Модальные окна -->
    <AIChat v-if="showAIChat" @close="showAIChat = false" />
    <DiagnosisModal 
      v-if="showDiagnosisModal" 
      :system-id="diagnosisSystemId"
      @close="showDiagnosisModal = false"
      @completed="onDiagnosisCompleted"
    />
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'
import { ragService } from '@/services/ragService'
import AIChat from '@/components/AIChat.vue'
import DiagnosisModal from '@/components/DiagnosisModal.vue'

export default {
  name: 'Dashboard',
  components: {
    AIChat,
    DiagnosisModal
  },
  setup() {
    const router = useRouter()
    
    // Реактивные данные
    const dashboardStats = ref({})
    const attentionSystems = ref([])
    const recentReports = ref([])
    const aiRecommendations = ref([])
    const availableSystems = ref([])
    const quickDiagnosisResult = ref(null)
    
    const isRefreshing = ref(false)
    const autoRefresh = ref(true)
    const autoRefreshInterval = ref(null)
    const selectedSystemForDiagnosis = ref('')
    const isDiagnosing = ref(false)
    const showAIChat = ref(false)
    const showDiagnosisModal = ref(false)
    const diagnosisSystemId = ref(null)
    
    // Вычисляемые свойства
    const overallHealthScore = computed(() => {
      if (!dashboardStats.value.user_systems) return 0
      
      const total = dashboardStats.value.user_systems.total || 0
      const active = dashboardStats.value.user_systems.active || 0
      const critical = dashboardStats.value.recent_activity?.systems_with_issues || 0
      
      if (total === 0) return 100
      
      const healthRatio = (active - critical) / total
      return Math.max(0, Math.round(healthRatio * 100))
    })
    
    const healthStatus = computed(() => {
      const score = overallHealthScore.value
      if (score >= 90) return 'Отлично'
      if (score >= 75) return 'Хорошо'
      if (score >= 60) return 'Удовлетворительно'
      if (score >= 40) return 'Требует внимания'
      return 'Критично'
    })
    
    const healthChangeClass = computed(() => {
      const score = overallHealthScore.value
      if (score >= 75) return 'positive'
      if (score >= 50) return 'neutral'
      return 'negative'
    })
    
    // Методы
    const loadDashboardData = async () => {
      try {
        isRefreshing.value = true
        
        // Загрузка статистики дашборда
        const stats = await hydraulicSystemService.getDashboardStats()
        dashboardStats.value = stats
        
        // Загрузка систем требующих внимания
        const attention = stats.systems_needing_attention || []
        
        // Дополнение данных о проблемах для каждой системы
        for (let system of attention) {
          try {
            const systemData = await hydraulicSystemService.getSystem(system.id)
            system.recent_issues = [
              {
                id: 1,
                type: 'critical',
                message: `Критических событий: ${system.critical_count}`,
                timestamp: new Date().toISOString()
              }
            ]
          } catch (error) {
            console.error(`Ошибка загрузки данных системы ${system.id}:`, error)
          }
        }
        
        attentionSystems.value = attention
        
        // Загрузка последних отчетов
        // TODO: Добавить API endpoint для последних отчетов
        recentReports.value = []
        
        // Загрузка доступных систем для диагностики
        const systemsResponse = await hydraulicSystemService.getSystems()
        const systems = Array.isArray(systemsResponse) ? systemsResponse : systemsResponse.results || []
        availableSystems.value = systems.filter(s => s.status === 'active').slice(0, 10)
        
      } catch (error) {
        console.error('Ошибка загрузки данных дашборда:', error)
      } finally {
        isRefreshing.value = false
      }
    }
    
    const generateRecommendations = async () => {
      try {
        // Получение рекомендаций от RAG системы на основе текущего состояния
        const symptoms = []
        
        if (dashboardStats.value.recent_activity?.critical_events > 0) {
          symptoms.push('критические события')
        }
        
        if (attentionSystems.value.length > 0) {
          symptoms.push('проблемы систем')
        }
        
        if (symptoms.length === 0) {
          symptoms.push('профилактическое обслуживание')
        }
        
        const response = await ragService.getRecommendations(symptoms)
        
        // Преобразование в формат рекомендаций дашборда
        aiRecommendations.value = response.recommendations.map((rec, index) => ({
          id: index + 1,
          title: rec.title || 'AI Рекомендация',
          description: rec.description || rec.content,
          priority: rec.priority || 'medium',
          category: rec.category || 'general'
        }))
        
      } catch (error) {
        console.error('Ошибка генерации рекомендаций:', error)
        // Заглушка при ошибке
        aiRecommendations.value = [
          {
            id: 1,
            title: 'Проверка фильтров',
            description: 'Рекомендуется проверить состояние фильтров гидросистем',
            priority: 'medium'
          }
        ]
      }
    }
    
    const runQuickDiagnosis = async () => {
      if (!selectedSystemForDiagnosis.value) return
      
      try {
        isDiagnosing.value = true
        quickDiagnosisResult.value = null
        
        const response = await hydraulicSystemService.runHealthCheck(selectedSystemForDiagnosis.value)
        quickDiagnosisResult.value = response
        
      } catch (error) {
        console.error('Ошибка быстрой диагностики:', error)
        quickDiagnosisResult.value = {
          health_score: 0,
          status: 'error',
          issues: ['Ошибка выполнения диагностики']
        }
      } finally {
        isDiagnosing.value = false
      }
    }
    
    const runDiagnosis = (systemId) => {
      diagnosisSystemId.value = systemId
      showDiagnosisModal.value = true
    }
    
    const refreshDashboard = () => {
      loadDashboardData()
    }
    
    const toggleAutoRefresh = () => {
      if (autoRefresh.value) {
        startAutoRefresh()
      } else {
        stopAutoRefresh()
      }
    }
    
    const startAutoRefresh = () => {
      stopAutoRefresh()
      autoRefreshInterval.value = setInterval(() => {
        if (!isRefreshing.value) {
          loadDashboardData()
        }
      }, 30000) // Обновление каждые 30 секунд
    }
    
    const stopAutoRefresh = () => {
      if (autoRefreshInterval.value) {
        clearInterval(autoRefreshInterval.value)
        autoRefreshInterval.value = null
      }
    }
    
    const navigateToSystem = (systemId) => {
      router.push(`/systems/${systemId}`)
    }
    
    const viewSystemDetails = (systemId) => {
      router.push(`/systems/${systemId}/details`)
    }
    
    const openAIChat = () => {
      showAIChat.value = true
    }
    
    const applyRecommendation = (recommendation) => {
      // TODO: Реализация применения рекомендации
      console.log('Применение рекомендации:', recommendation)
    }
    
    const dismissRecommendation = (recommendationId) => {
      aiRecommendations.value = aiRecommendations.value.filter(r => r.id !== recommendationId)
    }
    
    const openFullDiagnosisReport = () => {
      if (selectedSystemForDiagnosis.value) {
        router.push(`/systems/${selectedSystemForDiagnosis.value}/diagnostics`)
      }
    }
    
    const onDiagnosisCompleted = () => {
      showDiagnosisModal.value = false
      loadDashboardData() // Обновить данные после диагностики
    }
    
    // Вспомогательные функции
    const getIssueIcon = (type) => {
      const icons = {
        'critical': '🚨',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
      }
      return icons[type] || '❓'
    }
    
    const getSeverityIcon = (severity) => {
      const icons = {
        'critical': '🚨',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
      }
      return icons[severity] || 'ℹ️'
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
    
    const getStatusText = (status) => {
      const statusTexts = {
        'excellent': 'Отлично',
        'good': 'Хорошо',
        'fair': 'Удовлетворительно',
        'poor': 'Плохо',
        'critical': 'Критично',
        'error': 'Ошибка'
      }
      return statusTexts[status] || 'Неизвестно'
    }
    
    const formatTimeAgo = (timestamp) => {
      const now = new Date()
      const time = new Date(timestamp)
      const diffMinutes = Math.floor((now - time) / (1000 * 60))
      
      if (diffMinutes < 1) return 'только что'
      if (diffMinutes < 60) return `${diffMinutes} мин. назад`
      if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)} ч. назад`
      return `${Math.floor(diffMinutes / 1440)} дн. назад`
    }
    
    const formatDate = (dateString) => {
      return new Date(dateString).toLocaleDateString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        year: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    
    const truncateText = (text, length) => {
      if (!text) return ''
      return text.length > length ? text.substring(0, length) + '...' : text
    }
    
    // Жизненный цикл
    onMounted(() => {
      loadDashboardData()
      generateRecommendations()
      if (autoRefresh.value) {
        startAutoRefresh()
      }
    })
    
    onUnmounted(() => {
      stopAutoRefresh()
    })
    
    return {
      dashboardStats,
      attentionSystems,
      recentReports,
      aiRecommendations,
      availableSystems,
      quickDiagnosisResult,
      isRefreshing,
      autoRefresh,
      selectedSystemForDiagnosis,
      isDiagnosing,
      showAIChat,
      showDiagnosisModal,
      overallHealthScore,
      healthStatus,
      healthChangeClass,
      loadDashboardData,
      generateRecommendations,
      runQuickDiagnosis,
      runDiagnosis,
      refreshDashboard,
      toggleAutoRefresh,
      navigateToSystem,
      viewSystemDetails,
      openAIChat,
      applyRecommendation,
      dismissRecommendation,
      openFullDiagnosisReport,
      onDiagnosisCompleted,
      getIssueIcon,
      getSeverityIcon,
      getPriorityIcon,
      getStatusText,
      formatTimeAgo,
      formatDate,
      truncateText
    }
  }
}
</script>

<style scoped>
/* Основные стили дашборда */
.dashboard-container {
  padding: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

/* Заголовок дашборда */
.dashboard-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 2rem;
  border-radius: 0 0 20px 20px;
  margin-bottom: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.header-content h1 {
  font-size: 2.25rem;
  font-weight: 700;
  color: #2d3748;
  margin: 0 0 0.5rem 0;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.dashboard-subtitle {
  color: #718096;
  font-size: 1.125rem;
  margin: 0;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.refresh-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.refresh-icon.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Переключатель авто-обновления */
.auto-refresh-toggle {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.toggle-switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e0;
  transition: .4s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

input:checked + .slider:before {
  transform: translateX(26px);
}

.toggle-label {
  color: #4a5568;
  font-weight: 500;
}

/* Сетка метрик */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
  padding: 0 2rem;
}

.metric-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 2rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s;
}

.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

.metric-icon {
  font-size: 3rem;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(102, 126, 234, 0.1);
}

.metric-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #2d3748;
  line-height: 1;
  margin-bottom: 0.25rem;
}

.metric-label {
  color: #718096;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.metric-change {
  font-size: 0.875rem;
  font-weight: 600;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  display: inline-block;
}

.metric-change.positive {
  background: rgba(72, 187, 120, 0.1);
  color: #38a169;
}

.metric-change.negative {
  background: rgba(245, 101, 101, 0.1);
  color: #e53e3e;
}

.metric-change.neutral {
  background: rgba(237, 137, 54, 0.1);
  color: #dd6b20;
}

/* Основной контент */
.dashboard-main {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;
  padding: 0 2rem 2rem;
}

/* Секции дашборда */
.dashboard-section {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  margin-bottom: 2rem;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.section-header {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  background: rgba(102, 126, 234, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2d3748;
  margin: 0;
}

.section-count {
  background: #667eea;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.section-link {
  color: #667eea;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s;
}

.section-link:hover {
  color: #553c9a;
}

/* Системы требующие внимания */
.attention-systems {
  padding: 1.5rem;
}

.no-issues {
  text-align: center;
  padding: 3rem 1rem;
  color: #718096;
}

.no-issues-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.attention-system-card {
  background: #f7fafc;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.attention-system-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.system-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.system-name {
  font-weight: 600;
  color: #2d3748;
  font-size: 1.125rem;
}

.critical-badge {
  background: #fed7d7;
  color: #c53030;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.system-issues {
  margin-bottom: 1rem;
}

.issue-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.issue-type {
  font-size: 1.1rem;
}

.issue-text {
  flex: 1;
  color: #4a5568;
}

.issue-time {
  color: #a0aec0;
  font-size: 0.8rem;
}

.system-actions {
  display: flex;
  gap: 0.75rem;
}

.action-btn {
  flex: 1;
  padding: 0.75rem;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn.diagnose {
  background: linear-gradient(135deg, #4299e1, #3182ce);
  color: white;
}

.action-btn.view {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
}

.action-btn:hover {
  transform: translateY(-1px);
}

/* Отчеты */
.recent-reports {
  padding: 1.5rem;
}

.report-card {
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  border-left: 4px solid;
  transition: all 0.2s;
}

.report-card:hover {
  transform: translateX(4px);
}

.report-critical {
  background: #fed7d7;
  border-left-color: #f56565;
}

.report-error {
  background: #fed7d7;
  border-left-color: #f56565;
}

.report-warning {
  background: #fef5e7;
  border-left-color: #ed8936;
}

.report-info {
  background: #ebf8ff;
  border-left-color: #4299e1;
}

.report-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.report-severity {
  font-size: 1.25rem;
}

.report-info {
  flex: 1;
}

.report-title {
  font-weight: 600;
  color: #2d3748;
}

.report-system {
  color: #718096;
  font-size: 0.9rem;
}

.report-time {
  color: #a0aec0;
  font-size: 0.8rem;
}

.report-description {
  color: #4a5568;
  font-size: 0.9rem;
  line-height: 1.4;
}

/* Диаграмма состояния систем */
.systems-status-chart {
  padding: 2rem;
  text-align: center;
}

.chart-legend {
  margin-top: 2rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  justify-content: center;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 4px;
}

.legend-color.active { background: #48bb78; }
.legend-color.maintenance { background: #ed8936; }
.legend-color.inactive { background: #a0aec0; }

/* AI Рекомендации */
.ai-recommendations {
  padding: 1.5rem;
}

.no-recommendations {
  text-align: center;
  padding: 2rem 1rem;
  color: #718096;
}

.ai-thinking {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.generate-recommendations-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.generate-recommendations-btn:hover {
  transform: translateY(-2px);
}

.recommendation-card {
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  border: 1px solid;
  transition: all 0.2s;
}

.priority-urgent {
  background: #fed7d7;
  border-color: #f56565;
}

.priority-high {
  background: #fef5e7;
  border-color: #ed8936;
}

.priority-medium {
  background: #fefcbf;
  border-color: #f6e05e;
}

.priority-low {
  background: #f0fff4;
  border-color: #68d391;
}

.recommendation-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.priority-badge {
  font-size: 1.25rem;
}

.recommendation-title {
  font-weight: 600;
  color: #2d3748;
}

.recommendation-description {
  color: #4a5568;
  margin-bottom: 1rem;
  line-height: 1.5;
}

.recommendation-actions {
  display: flex;
  gap: 0.75rem;
}

.recommendation-action,
.recommendation-dismiss {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.recommendation-action {
  background: #48bb78;
  color: white;
}

.recommendation-dismiss {
  background: #e2e8f0;
  color: #4a5568;
}

.ai-chat-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.ai-chat-btn:hover {
  transform: translateY(-1px);
}

/* Быстрая диагностика */
.quick-diagnostics {
  padding: 1.5rem;
}

.diagnostic-selector {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.system-select {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  color: #4a5568;
}

.diagnose-btn {
  background: linear-gradient(135deg, #4299e1, #3182ce);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.diagnose-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.diagnosis-result {
  background: #f7fafc;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.result-score {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2d3748;
}

.result-status {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.status-excellent { background: #c6f6d5; color: #22543d; }
.status-good { background: #bee3f8; color: #2a69ac; }
.status-fair { background: #fef5e7; color: #c05621; }
.status-poor { background: #fed7d7; color: #c53030; }
.status-critical { background: #fed7d7; color: #c53030; }

.result-issues {
  margin-bottom: 1.5rem;
}

.issues-title {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.issues-list {
  margin: 0;
  padding-left: 1.5rem;
  color: #4a5568;
}

.full-report-btn {
  background: #667eea;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.full-report-btn:hover {
  background: #553c9a;
}

/* Адаптивность */
@media (max-width: 1200px) {
  .dashboard-main {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .dashboard-header {
    flex-direction: column;
    gap: 1.5rem;
    align-items: stretch;
    text-align: center;
  }
  
  .header-actions {
    justify-content: center;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr;
    padding: 0 1rem;
  }
  
  .dashboard-main {
    padding: 0 1rem 1rem;
  }
  
  .metric-card {
    padding: 1.5rem;
  }
  
  .metric-icon {
    font-size: 2.5rem;
    width: 50px;
    height: 50px;
  }
  
  .metric-value {
    font-size: 2rem;
  }
}

@media (max-width: 480px) {
  .dashboard-header {
    padding: 1.5rem;
  }
  
  .header-content h1 {
    font-size: 1.75rem;
  }
  
  .metrics-grid,
  .dashboard-main {
    padding: 0 0.75rem;
  }
  
  .system-actions {
    flex-direction: column;
  }
  
  .diagnostic-selector {
    flex-direction: column;
  }
}
</style>
