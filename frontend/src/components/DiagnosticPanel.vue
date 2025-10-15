<template>
  <div class="diagnostic-panel">
    <!-- Заголовок панели -->
    <div class="panel-header">
      <div class="header-info">
        <h2 class="panel-title">
          🔍 AI Диагностика
        </h2>
        <p class="panel-subtitle">
          {{ systemName ? `Система: ${systemName}` : 'Комплексная диагностика систем' }}
        </p>
      </div>
      
      <div class="header-actions">
        <button 
          class="action-btn refresh-btn" 
          @click="refreshDiagnostic"
          :disabled="isAnalyzing"
        >
          <span class="btn-icon" :class="{ spinning: isAnalyzing }">🔄</span>
          {{ isAnalyzing ? 'Анализ...' : 'Обновить' }}
        </button>
        
        <button class="action-btn export-btn" @click="exportResults">
          📊 Экспорт
        </button>
      </div>
    </div>

    <!-- Статус диагностики -->
    <div class="diagnostic-status" :class="`status-${currentStatus}`">
      <div class="status-icon">{{ getStatusIcon(currentStatus) }}</div>
      <div class="status-info">
        <div class="status-title">{{ getStatusTitle(currentStatus) }}</div>
        <div class="status-description">{{ getStatusDescription(currentStatus) }}</div>
      </div>
      <div class="status-actions" v-if="currentStatus === 'completed'">
        <div class="confidence-score">
          Уверенность: {{ Math.round((diagnosticResult?.analysis_confidence || 0) * 100) }}%
        </div>
      </div>
    </div>

    <!-- Основной контент -->
    <div class="diagnostic-content" v-if="!isAnalyzing">
      
      <!-- Результаты диагностики -->
      <div v-if="diagnosticResult" class="diagnostic-results">
        
        <!-- Общее состояние системы -->
        <div class="result-section system-health">
          <h3 class="section-title">❤️ Общее состояние системы</h3>
          
          <div class="health-overview">
            <div class="health-score">
              <div class="score-circle" :class="`score-${getHealthLevel(systemHealth.score)}`">
                <div class="score-value">{{ systemHealth.score || 0 }}%</div>
                <div class="score-label">Здоровье</div>
              </div>
            </div>
            
            <div class="health-details">
              <div class="health-item">
                <span class="health-label">Статус:</span>
                <span class="health-value" :class="`status-${systemHealth.status}`">
                  {{ systemHealth.status_text || 'Неизвестно' }}
                </span>
              </div>
              
              <div class="health-item">
                <span class="health-label">Последнее обновление:</span>
                <span class="health-value">{{ formatDateTime(systemHealth.last_updated) }}</span>
              </div>
              
              <div class="health-item" v-if="diagnosticResult.data_points_processed">
                <span class="health-label">Обработано точек данных:</span>
                <span class="health-value">{{ diagnosticResult.data_points_processed }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Обнаруженные аномалии -->
        <div class="result-section anomalies-section">
          <h3 class="section-title">⚠️ Обнаруженные аномалии</h3>
          
          <div class="anomalies-overview">
            <div class="anomaly-score">
              <div class="score-indicator" :class="getAnomalyLevel(anomalies.anomaly_score)">
                {{ Math.round((anomalies.anomaly_score || 0) * 100) }}%
              </div>
              <div class="score-description">Уровень аномальности</div>
            </div>
            
            <div class="risk-level">
              <div class="risk-badge" :class="`risk-${anomalies.risk_level}`">
                {{ getRiskLevelText(anomalies.risk_level) }}
              </div>
            </div>
          </div>
          
          <div class="anomalies-list" v-if="anomalies.anomalies && anomalies.anomalies.length > 0">
            <div 
              v-for="(anomaly, index) in anomalies.anomalies" 
              :key="index"
              class="anomaly-item"
              :class="`severity-${anomaly.severity}`"
            >
              <div class="anomaly-header">
                <div class="anomaly-sensor">
                  {{ getSensorDisplayName(anomaly.sensor_type) }}
                </div>
                <div class="anomaly-value">
                  {{ anomaly.value }} {{ getSensorUnit(anomaly.sensor_type) }}
                </div>
                <div class="anomaly-severity">
                  {{ getSeverityIcon(anomaly.severity) }}
                </div>
              </div>
              <div class="anomaly-message">
                {{ anomaly.message }}
              </div>
            </div>
          </div>
          
          <div v-else class="no-anomalies">
            <div class="no-anomalies-icon">✅</div>
            <p>Аномалии не обнаружены. Система работает в штатном режиме.</p>
          </div>
        </div>

        <!-- Прогнозы отказов -->
        <div class="result-section predictions-section">
          <h3 class="section-title">🔮 Прогнозы отказов</h3>
          
          <div class="predictions-overview">
            <div class="failure-probability">
              <div class="probability-gauge">
                <div class="gauge-fill" :style="{ width: (predictions.failure_probability * 100) + '%' }"></div>
              </div>
              <div class="probability-text">
                {{ Math.round((predictions.failure_probability || 0) * 100) }}% вероятность отказа
              </div>
            </div>
            
            <div class="maintenance-urgency">
              <div class="urgency-badge" :class="`urgency-${predictions.maintenance_urgency}`">
                {{ getUrgencyText(predictions.maintenance_urgency) }}
              </div>
            </div>
          </div>
          
          <div class="predictions-list" v-if="predictions.predictions && predictions.predictions.length > 0">
            <div 
              v-for="(prediction, index) in predictions.predictions" 
              :key="index"
              class="prediction-item"
            >
              <div class="prediction-header">
                <div class="prediction-component">{{ prediction.component }}</div>
                <div class="prediction-probability">
                  {{ Math.round(prediction.probability * 100) }}%
                </div>
              </div>
              
              <div class="prediction-details">
                <div class="prediction-type">
                  <strong>Тип отказа:</strong> {{ getFailureTypeText(prediction.failure_type) }}
                </div>
                <div class="prediction-time">
                  <strong>Ожидаемое время:</strong> {{ prediction.time_to_failure }}
                </div>
                <div class="prediction-description">{{ prediction.description }}</div>
              </div>
              
              <div class="prediction-action">
                <button 
                  class="action-recommendation-btn"
                  @click="showRecommendationDetails(prediction)"
                >
                  💡 {{ prediction.recommended_action }}
                </button>
              </div>
            </div>
          </div>
          
          <div v-else class="no-predictions">
            <div class="no-predictions-icon">🎯</div>
            <p>Критических прогнозов нет. Система работает стабильно.</p>
          </div>
        </div>

        <!-- AI Рекомендации -->
        <div class="result-section recommendations-section">
          <h3 class="section-title">🤖 AI Рекомендации</h3>
          
          <div class="recommendations-list" v-if="recommendations && recommendations.length > 0">
            <div 
              v-for="(recommendation, index) in recommendations" 
              :key="index"
              class="recommendation-item"
              :class="`priority-${recommendation.priority}`"
            >
              <div class="recommendation-header">
                <div class="recommendation-icon">
                  {{ getPriorityIcon(recommendation.priority) }}
                </div>
                <div class="recommendation-info">
                  <div class="recommendation-title">{{ recommendation.title }}</div>
                  <div class="recommendation-category">{{ getCategoryText(recommendation.category) }}</div>
                </div>
                <div class="recommendation-priority">
                  {{ getPriorityText(recommendation.priority) }}
                </div>
              </div>
              
              <div class="recommendation-content">
                <div class="recommendation-description">
                  {{ recommendation.description }}
                </div>
                
                <div class="recommendation-action">
                  <strong>Действие:</strong> {{ recommendation.action }}
                </div>
                
                <div class="recommendation-meta">
                  <div class="recommendation-time">
                    <span class="meta-label">Время:</span>
                    <span class="meta-value">{{ recommendation.estimated_time }}</span>
                  </div>
                  <div class="recommendation-cost">
                    <span class="meta-label">Стоимость:</span>
                    <span class="meta-value">{{ recommendation.cost_estimate }}</span>
                  </div>
                  <div class="recommendation-impact">
                    <span class="meta-label">Влияние на безопасность:</span>
                    <span class="meta-value">{{ recommendation.safety_impact }}</span>
                  </div>
                </div>
              </div>
              
              <div class="recommendation-actions">
                <button 
                  class="recommendation-btn apply"
                  @click="applyRecommendation(recommendation)"
                >
                  ✅ Принять
                </button>
                <button 
                  class="recommendation-btn schedule"
                  @click="scheduleRecommendation(recommendation)"
                >
                  📅 Запланировать
                </button>
                <button 
                  class="recommendation-btn dismiss"
                  @click="dismissRecommendation(index)"
                >
                  ❌ Отклонить
                </button>
              </div>
            </div>
          </div>
          
          <div v-else class="no-recommendations">
            <div class="no-recommendations-icon">🎯</div>
            <p>Специальных рекомендаций нет. Система функционирует нормально.</p>
          </div>
        </div>

        <!-- Краткая сводка -->
        <div class="result-section summary-section">
          <h3 class="section-title">📋 Краткая сводка</h3>
          
          <div class="summary-content">
            <div class="summary-text">
              {{ diagnosticResult.summary || 'Анализ завершен без критических замечаний.' }}
            </div>
            
            <div class="summary-stats">
              <div class="stat-item">
                <span class="stat-value">{{ diagnosticResult.features_analyzed || 0 }}</span>
                <span class="stat-label">Признаков проанализировано</span>
              </div>
              
              <div class="stat-item">
                <span class="stat-value">{{ formatDateTime(diagnosticResult.timestamp) }}</span>
                <span class="stat-label">Время анализа</span>
              </div>
              
              <div class="stat-item" v-if="diagnosticResult.analysis_confidence">
                <span class="stat-value">{{ Math.round(diagnosticResult.analysis_confidence * 100) }}%</span>
                <span class="stat-label">Уверенность AI</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Отсутствие результатов -->
      <div v-else class="no-results">
        <div class="no-results-icon">🔍</div>
        <h3>Диагностика не проводилась</h3>
        <p>Запустите диагностику для получения детального анализа состояния системы.</p>
        <button class="start-diagnostic-btn" @click="startDiagnostic">
          🚀 Запустить диагностику
        </button>
      </div>
    </div>

    <!-- Индикатор загрузки -->
    <div v-else class="diagnostic-loading">
      <div class="loading-animation">
        <div class="loading-brain">🧠</div>
        <div class="loading-waves">
          <div class="wave"></div>
          <div class="wave"></div>
          <div class="wave"></div>
        </div>
      </div>
      
      <div class="loading-text">
        <h3>{{ loadingStage }}</h3>
        <p>{{ loadingDescription }}</p>
      </div>
      
      <div class="loading-progress">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: loadingProgress + '%' }"></div>
        </div>
        <div class="progress-text">{{ loadingProgress }}%</div>
      </div>
    </div>

    <!-- Модальные окна -->
    <RecommendationDetailsModal 
      v-if="showRecommendationModal"
      :recommendation="selectedRecommendation"
      @close="showRecommendationModal = false"
      @apply="applyRecommendation"
    />
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'
import RecommendationDetailsModal from './RecommendationDetailsModal.vue'

export default {
  name: 'DiagnosticPanel',
  components: {
    RecommendationDetailsModal
  },
  props: {
    systemId: {
      type: [String, Number],
      required: false
    },
    systemName: {
      type: String,
      default: ''
    }
  },
  setup(props) {
    // Реактивные данные
    const isAnalyzing = ref(false)
    const currentStatus = ref('idle') // idle, running, completed, error
    const diagnosticResult = ref(null)
    const loadingStage = ref('Инициализация...')
    const loadingDescription = ref('Подготовка к анализу данных')
    const loadingProgress = ref(0)
    const showRecommendationModal = ref(false)
    const selectedRecommendation = ref(null)
    
    // Вычисляемые свойства для удобства доступа к данным
    const systemHealth = computed(() => diagnosticResult.value?.system_health || {})
    const anomalies = computed(() => diagnosticResult.value?.anomalies || {})
    const predictions = computed(() => diagnosticResult.value?.predictions || {})
    const recommendations = computed(() => diagnosticResult.value?.recommendations || [])
    
    // Методы
    const startDiagnostic = async () => {
      if (!props.systemId) {
        alert('Выберите систему для диагностики')
        return
      }
      
      isAnalyzing.value = true
      currentStatus.value = 'running'
      loadingProgress.value = 0
      
      try {
        // Имитация этапов диагностики
        const stages = [
          { text: 'Сбор данных датчиков...', description: 'Получение актуальных показаний', duration: 2000 },
          { text: 'Анализ трендов...', description: 'Выявление закономерностей в данных', duration: 3000 },
          { text: 'Обнаружение аномалий...', description: 'Поиск отклонений от нормы', duration: 2500 },
          { text: 'Прогнозирование отказов...', description: 'AI анализ вероятных проблем', duration: 3500 },
          { text: 'Генерация рекомендаций...', description: 'Подготовка советов по обслуживанию', duration: 2000 },
          { text: 'Финализация отчета...', description: 'Сборка результатов анализа', duration: 1000 }
        ]
        
        let totalProgress = 0
        const progressStep = 100 / stages.length
        
        for (const stage of stages) {
          loadingStage.value = stage.text
          loadingDescription.value = stage.description
          
          await new Promise(resolve => {
            const startProgress = totalProgress
            const interval = setInterval(() => {
              totalProgress = Math.min(totalProgress + 2, startProgress + progressStep)
              loadingProgress.value = Math.round(totalProgress)
              
              if (totalProgress >= startProgress + progressStep) {
                clearInterval(interval)
                resolve()
              }
            }, stage.duration / (progressStep / 2))
          })
        }
        
        // Запуск реальной диагностики
        const response = await hydraulicSystemService.runDiagnosis(props.systemId)
        diagnosticResult.value = response
        currentStatus.value = 'completed'
        
      } catch (error) {
        console.error('Ошибка диагностики:', error)
        currentStatus.value = 'error'
      } finally {
        isAnalyzing.value = false
        loadingProgress.value = 100
      }
    }
    
    const refreshDiagnostic = () => {
      startDiagnostic()
    }
    
    const exportResults = () => {
      if (!diagnosticResult.value) {
        alert('Нет данных для экспорта')
        return
      }
      
      const exportData = {
        system_id: props.systemId,
        system_name: props.systemName,
        diagnostic_result: diagnosticResult.value,
        exported_at: new Date().toISOString()
      }
      
      const dataStr = JSON.stringify(exportData, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      
      const link = document.createElement('a')
      link.href = URL.createObjectURL(dataBlob)
      link.download = `diagnostic_${props.systemId}_${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    
    const showRecommendationDetails = (recommendation) => {
      selectedRecommendation.value = recommendation
      showRecommendationModal.value = true
    }
    
    const applyRecommendation = (recommendation) => {
      // Реализация применения рекомендации
      console.log('Применение рекомендации:', recommendation)
      alert(`Рекомендация "${recommendation.title}" принята к исполнению`)
      showRecommendationModal.value = false
    }
    
    const scheduleRecommendation = (recommendation) => {
      // Реализация планирования рекомендации
      console.log('Планирование рекомендации:', recommendation)
      alert(`Рекомендация "${recommendation.title}" добавлена в план обслуживания`)
    }
    
    const dismissRecommendation = (index) => {
      if (confirm('Отклонить эту рекомендацию?')) {
        const newRecommendations = [...recommendations.value]
        newRecommendations.splice(index, 1)
        diagnosticResult.value.recommendations = newRecommendations
      }
    }
    
    // Вспомогательные функции для отображения
    const getStatusIcon = (status) => {
      const icons = {
        'idle': '⏳',
        'running': '🔄',
        'completed': '✅',
        'error': '❌'
      }
      return icons[status] || '❓'
    }
    
    const getStatusTitle = (status) => {
      const titles = {
        'idle': 'Готов к диагностике',
        'running': 'Выполняется анализ',
        'completed': 'Диагностика завершена',
        'error': 'Ошибка диагностики'
      }
      return titles[status] || 'Неизвестное состояние'
    }
    
    const getStatusDescription = (status) => {
      const descriptions = {
        'idle': 'Нажмите кнопку для запуска анализа системы',
        'running': 'AI анализирует данные датчиков и выявляет проблемы',
        'completed': 'Анализ успешно завершен, результаты готовы',
        'error': 'Произошла ошибка при выполнении диагностики'
      }
      return descriptions[status] || ''
    }
    
    const getHealthLevel = (score) => {
      if (score >= 90) return 'excellent'
      if (score >= 75) return 'good'
      if (score >= 60) return 'fair'
      if (score >= 40) return 'poor'
      return 'critical'
    }
    
    const getAnomalyLevel = (score) => {
      if (score <= 0.2) return 'low'
      if (score <= 0.5) return 'medium'
      if (score <= 0.8) return 'high'
      return 'critical'
    }
    
    const getRiskLevelText = (level) => {
      const texts = {
        'minimal': 'Минимальный риск',
        'low': 'Низкий риск',
        'medium': 'Средний риск',
        'high': 'Высокий риск',
        'critical': 'Критический риск'
      }
      return texts[level] || level
    }
    
    const getSensorDisplayName = (sensor) => {
      const names = {
        'pressure': 'Давление',
        'temperature': 'Температура',
        'flow': 'Расход',
        'vibration': 'Вибрация'
      }
      return names[sensor] || sensor
    }
    
    const getSensorUnit = (sensor) => {
      const units = {
        'pressure': 'бар',
        'temperature': '°C',
        'flow': 'л/мин',
        'vibration': 'мм/с'
      }
      return units[sensor] || ''
    }
    
    const getSeverityIcon = (severity) => {
      const icons = {
        'critical': '🚨',
        'warning': '⚠️',
        'info': 'ℹ️'
      }
      return icons[severity] || '❓'
    }
    
    const getUrgencyText = (urgency) => {
      const texts = {
        'immediate': 'Немедленно',
        'urgent': 'Срочно',
        'planned': 'Плановое',
        'routine': 'Рутинное'
      }
      return texts[urgency] || urgency
    }
    
    const getFailureTypeText = (type) => {
      const texts = {
        'overload': 'Перегрузка',
        'underperformance': 'Снижение производительности',
        'gradual_degradation': 'Постепенная деградация',
        'multiple_system_degradation': 'Комплексная деградация'
      }
      return texts[type] || type
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
    
    const getPriorityText = (priority) => {
      const texts = {
        'urgent': 'Критично',
        'high': 'Высокий',
        'medium': 'Средний',
        'low': 'Низкий'
      }
      return texts[priority] || priority
    }
    
    const getCategoryText = (category) => {
      const texts = {
        'maintenance': 'Техобслуживание',
        'monitoring': 'Мониторинг',
        'predictive_maintenance': 'Прогнозное обслуживание',
        'routine_maintenance': 'Плановое обслуживание',
        'system_check': 'Проверка системы'
      }
      return texts[category] || category
    }
    
    const formatDateTime = (dateTime) => {
      if (!dateTime) return 'Не указано'
      return new Date(dateTime).toLocaleString('ru-RU')
    }
    
    // Наблюдатели
    watch(() => props.systemId, (newId) => {
      if (newId) {
        diagnosticResult.value = null
        currentStatus.value = 'idle'
      }
    })
    
    // Жизненный цикл
    onMounted(() => {
      // Автозапуск диагностики если передан systemId
      if (props.systemId) {
        // startDiagnostic()
      }
    })
    
    return {
      isAnalyzing,
      currentStatus,
      diagnosticResult,
      loadingStage,
      loadingDescription,
      loadingProgress,
      showRecommendationModal,
      selectedRecommendation,
      systemHealth,
      anomalies,
      predictions,
      recommendations,
      startDiagnostic,
      refreshDiagnostic,
      exportResults,
      showRecommendationDetails,
      applyRecommendation,
      scheduleRecommendation,
      dismissRecommendation,
      getStatusIcon,
      getStatusTitle,
      getStatusDescription,
      getHealthLevel,
      getAnomalyLevel,
      getRiskLevelText,
      getSensorDisplayName,
      getSensorUnit,
      getSeverityIcon,
      getUrgencyText,
      getFailureTypeText,
      getPriorityIcon,
      getPriorityText,
      getCategoryText,
      formatDateTime
    }
  }
}
</script>

<style scoped>
/* Основные стили диагностической панели */
.diagnostic-panel {
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

/* Заголовок панели */
.panel-header {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.header-info h2 {
  margin: 0 0 0.5rem 0;
  font-size: 1.75rem;
  font-weight: 700;
}

.panel-subtitle {
  margin: 0;
  opacity: 0.9;
  font-size: 1rem;
}

.header-actions {
  display: flex;
  gap: 1rem;
}

.action-btn {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.action-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.btn-icon.spinning {
  animation: spin 1s linear infinite;
}

/* Статус диагностики */
.diagnostic-status {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.status-idle {
  background: linear-gradient(135deg, #edf2f7, #e2e8f0);
}

.status-running {
  background: linear-gradient(135deg, #fef5e7, #fed7aa);
  animation: pulse 2s infinite;
}

.status-completed {
  background: linear-gradient(135deg, #f0fff4, #c6f6d5);
}

.status-error {
  background: linear-gradient(135deg, #fed7d7, #feb2b2);
}

.status-icon {
  font-size: 2.5rem;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.7);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.status-info {
  flex: 1;
}

.status-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.status-description {
  color: #4a5568;
  font-size: 0.95rem;
}

.status-actions {
  text-align: right;
}

.confidence-score {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  padding: 0.5rem 1rem;
  border-radius: 12px;
  font-weight: 600;
}

/* Основной контент */
.diagnostic-content {
  padding: 2rem;
}

.result-section {
  background: white;
  border-radius: 12px;
  margin-bottom: 2rem;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.section-title {
  background: #f8fafc;
  padding: 1.5rem 2rem;
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #2d3748;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

/* Общее состояние системы */
.health-overview {
  padding: 2rem;
  display: flex;
  align-items: center;
  gap: 3rem;
}

.health-score {
  flex-shrink: 0;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  border: 6px solid;
}

.score-excellent { 
  border-color: #48bb78;
  background: linear-gradient(135deg, #f0fff4, #c6f6d5);
}
.score-good { 
  border-color: #4299e1;
  background: linear-gradient(135deg, #ebf8ff, #bee3f8);
}
.score-fair { 
  border-color: #ed8936;
  background: linear-gradient(135deg, #fef5e7, #fbd38d);
}
.score-poor { 
  border-color: #f56565;
  background: linear-gradient(135deg, #fed7d7, #feb2b2);
}
.score-critical { 
  border-color: #e53e3e;
  background: linear-gradient(135deg, #fed7d7, #fc8181);
}

.score-value {
  font-size: 2rem;
  font-weight: 700;
  color: #2d3748;
}

.score-label {
  font-size: 0.875rem;
  color: #718096;
  margin-top: 0.25rem;
}

.health-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.health-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid #f1f5f9;
}

.health-item:last-child {
  border-bottom: none;
}

.health-label {
  color: #4a5568;
  font-weight: 500;
}

.health-value {
  font-weight: 600;
  color: #2d3748;
}

.health-value.status-excellent { color: #38a169; }
.health-value.status-good { color: #3182ce; }
.health-value.status-fair { color: #dd6b20; }
.health-value.status-poor { color: #e53e3e; }
.health-value.status-critical { color: #c53030; }

/* Аномалии */
.anomalies-overview {
  padding: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f8fafc;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.anomaly-score {
  text-align: center;
}

.score-indicator {
  font-size: 2.5rem;
  font-weight: 700;
  padding: 1rem;
  border-radius: 12px;
  margin-bottom: 0.5rem;
}

.score-indicator.low {
  background: #c6f6d5;
  color: #22543d;
}
.score-indicator.medium {
  background: #fef5e7;
  color: #c05621;
}
.score-indicator.high {
  background: #fed7d7;
  color: #c53030;
}
.score-indicator.critical {
  background: #fed7d7;
  color: #742a2a;
}

.score-description {
  color: #4a5568;
  font-size: 0.875rem;
}

.risk-level {
  text-align: center;
}

.risk-badge {
  padding: 0.75rem 1.5rem;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.875rem;
}

.risk-minimal { background: #c6f6d5; color: #22543d; }
.risk-low { background: #bee3f8; color: #2a69ac; }
.risk-medium { background: #fef5e7; color: #c05621; }
.risk-high { background: #fed7d7; color: #c53030; }
.risk-critical { background: #fed7d7; color: #742a2a; }

.anomalies-list,
.predictions-list,
.recommendations-list {
  padding: 1.5rem;
}

.anomaly-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  border-left: 4px solid;
}

.severity-critical { border-left-color: #f56565; }
.severity-warning { border-left-color: #ed8936; }
.severity-info { border-left-color: #4299e1; }

.anomaly-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.anomaly-sensor {
  font-weight: 600;
  color: #2d3748;
}

.anomaly-value {
  background: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-weight: 600;
  color: #4a5568;
  border: 1px solid #e2e8f0;
}

.anomaly-severity {
  font-size: 1.25rem;
}

.anomaly-message {
  color: #4a5568;
  line-height: 1.5;
}

/* Прогнозы */
.predictions-overview {
  padding: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f8fafc;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.failure-probability {
  flex: 1;
  max-width: 400px;
}

.probability-gauge {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.gauge-fill {
  height: 100%;
  background: linear-gradient(90deg, #48bb78, #ed8936, #f56565);
  transition: width 0.5s ease-in-out;
}

.probability-text {
  color: #4a5568;
  font-weight: 500;
}

.urgency-badge {
  padding: 0.75rem 1.5rem;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.875rem;
}

.urgency-immediate { background: #fed7d7; color: #c53030; }
.urgency-urgent { background: #fef5e7; color: #c05621; }
.urgency-planned { background: #bee3f8; color: #2a69ac; }
.urgency-routine { background: #c6f6d5; color: #22543d; }

.prediction-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  border: 1px solid #e2e8f0;
}

.prediction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.prediction-component {
  font-weight: 600;
  color: #2d3748;
  font-size: 1.125rem;
}

.prediction-probability {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-weight: 600;
}

.prediction-details {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
  color: #4a5568;
}

.prediction-action {
  text-align: center;
}

.action-recommendation-btn {
  background: linear-gradient(135deg, #4299e1, #3182ce);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.action-recommendation-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
}

/* Рекомендации */
.recommendation-item {
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid;
  background: white;
}

.priority-urgent {
  border-color: #f56565;
  background: linear-gradient(135deg, #fed7d7, #feb2b2);
}
.priority-high {
  border-color: #ed8936;
  background: linear-gradient(135deg, #fef5e7, #fbd38d);
}
.priority-medium {
  border-color: #f6e05e;
  background: linear-gradient(135deg, #fefcbf, #faf089);
}
.priority-low {
  border-color: #68d391;
  background: linear-gradient(135deg, #f0fff4, #c6f6d5);
}

.recommendation-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.recommendation-icon {
  font-size: 1.5rem;
}

.recommendation-info {
  flex: 1;
}

.recommendation-title {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.recommendation-category {
  color: #718096;
  font-size: 0.875rem;
}

.recommendation-priority {
  background: rgba(255, 255, 255, 0.8);
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #4a5568;
}

.recommendation-content {
  margin-bottom: 1.5rem;
}

.recommendation-description {
  color: #4a5568;
  line-height: 1.6;
  margin-bottom: 1rem;
}

.recommendation-action {
  color: #2d3748;
  font-weight: 500;
  margin-bottom: 1rem;
}

.recommendation-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.75rem;
  background: rgba(255, 255, 255, 0.7);
  padding: 1rem;
  border-radius: 6px;
}

.meta-label {
  color: #718096;
  font-size: 0.8rem;
  font-weight: 500;
}

.meta-value {
  color: #2d3748;
  font-weight: 600;
  margin-left: 0.5rem;
}

.recommendation-actions {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.recommendation-btn {
  flex: 1;
  min-width: 120px;
  padding: 0.75rem;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.recommendation-btn.apply {
  background: #48bb78;
  color: white;
}

.recommendation-btn.schedule {
  background: #4299e1;
  color: white;
}

.recommendation-btn.dismiss {
  background: #e2e8f0;
  color: #4a5568;
}

.recommendation-btn:hover {
  transform: translateY(-2px);
}

/* Пустые состояния */
.no-anomalies,
.no-predictions,
.no-recommendations,
.no-results {
  text-align: center;
  padding: 3rem 2rem;
  color: #4a5568;
}

.no-anomalies-icon,
.no-predictions-icon,
.no-recommendations-icon,
.no-results-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.start-diagnostic-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-size: 1.125rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  margin-top: 1rem;
}

.start-diagnostic-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Сводка */
.summary-content {
  padding: 2rem;
}

.summary-text {
  background: #f8fafc;
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid #667eea;
  color: #2d3748;
  line-height: 1.6;
  margin-bottom: 2rem;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
}

.stat-item {
  text-align: center;
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.stat-label {
  color: #718096;
  font-size: 0.875rem;
}

/* Загрузка */
.diagnostic-loading {
  padding: 4rem 2rem;
  text-align: center;
}

.loading-animation {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 2rem;
}

.loading-brain {
  font-size: 4rem;
  margin-bottom: 1rem;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

.loading-waves {
  display: flex;
  gap: 0.5rem;
}

.wave {
  width: 12px;
  height: 12px;
  background: #667eea;
  border-radius: 50%;
  animation: wave 1.4s infinite ease-in-out;
}

.wave:nth-child(2) {
  animation-delay: 0.2s;
}

.wave:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes wave {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.loading-text h3 {
  color: #2d3748;
  margin: 0 0 0.5rem 0;
}

.loading-text p {
  color: #4a5568;
  margin: 0 0 2rem 0;
}

.loading-progress {
  max-width: 400px;
  margin: 0 auto;
}

.progress-bar {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  transition: width 0.3s ease-in-out;
}

.progress-text {
  color: #4a5568;
  font-weight: 600;
}

/* Адаптивность */
@media (max-width: 768px) {
  .panel-header {
    flex-direction: column;
    align-items: stretch;
    text-align: center;
    padding: 1.5rem;
  }
  
  .header-actions {
    justify-content: center;
  }
  
  .diagnostic-status {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
  
  .health-overview {
    flex-direction: column;
    gap: 2rem;
    text-align: center;
  }
  
  .anomalies-overview,
  .predictions-overview {
    flex-direction: column;
    gap: 1.5rem;
    text-align: center;
  }
  
  .recommendation-actions {
    flex-direction: column;
  }
  
  .recommendation-btn {
    min-width: auto;
  }
  
  .summary-stats {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  .diagnostic-content {
    padding: 1rem;
  }
  
  .section-title {
    padding: 1rem 1.5rem;
    font-size: 1.125rem;
  }
  
  .health-overview,
  .anomalies-list,
  .predictions-list,
  .recommendations-list {
    padding: 1rem;
  }
  
  .score-circle {
    width: 100px;
    height: 100px;
  }
  
  .score-value {
    font-size: 1.75rem;
  }
  
  .recommendation-meta {
    grid-template-columns: 1fr;
  }
}
</style>