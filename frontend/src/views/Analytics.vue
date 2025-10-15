<template>
  <div class="analytics-container">
    <!-- Заголовок аналитики -->
    <div class="analytics-header">
      <div class="header-content">
        <h1 class="analytics-title">
          📊 Аналитика систем
        </h1>
        <p class="analytics-subtitle">
          Детальный анализ производительности и состояния гидравлических систем
        </p>
      </div>
      
      <div class="header-controls">
        <!-- Выбор системы -->
        <div class="control-group">
          <label>Система:</label>
          <select v-model="selectedSystem" @change="loadAnalytics" class="system-selector">
            <option value="">Все системы</option>
            <option v-for="system in availableSystems" :key="system.id" :value="system.id">
              {{ system.name }}
            </option>
          </select>
        </div>
        
        <!-- Выбор периода -->
        <div class="control-group">
          <label>Период:</label>
          <select v-model="selectedPeriod" @change="loadAnalytics" class="period-selector">
            <option value="7">7 дней</option>
            <option value="30">30 дней</option>
            <option value="90">90 дней</option>
            <option value="365">1 год</option>
          </select>
        </div>
        
        <!-- Экспорт данных -->
        <button class="export-btn" @click="exportData">
          📤 Экспорт
        </button>
      </div>
    </div>

    <!-- Загрузка -->
    <div v-if="isLoading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>Загрузка аналитических данных...</p>
    </div>

    <!-- Основной контент аналитики -->
    <div v-else class="analytics-content">
      
      <!-- Ключевые метрики -->
      <div class="key-metrics">
        <div class="metric-card total-systems">
          <div class="metric-icon">⚙️</div>
          <div class="metric-data">
            <div class="metric-value">{{ analyticsData.total_data_points || 0 }}</div>
            <div class="metric-label">Точек данных</div>
          </div>
        </div>
        
        <div class="metric-card avg-health">
          <div class="metric-icon">❤️</div>
          <div class="metric-data">
            <div class="metric-value">{{ averageHealth }}%</div>
            <div class="metric-label">Средн. здоровье</div>
          </div>
        </div>
        
        <div class="metric-card critical-events">
          <div class="metric-icon">⚠️</div>
          <div class="metric-data">
            <div class="metric-value">{{ totalCriticalEvents }}</div>
            <div class="metric-label">Крит. событий</div>
          </div>
        </div>
        
        <div class="metric-card uptime">
          <div class="metric-icon">⏱️</div>
          <div class="metric-data">
            <div class="metric-value">{{ averageUptime }}%</div>
            <div class="metric-label">Время работы</div>
          </div>
        </div>
      </div>

      <!-- Графики -->
      <div class="charts-grid">
        
        <!-- График трендов датчиков -->
        <div class="chart-section">
          <div class="chart-header">
            <h3>📈 Тренды датчиков</h3>
            <div class="chart-controls">
              <div class="sensor-filters">
                <label v-for="sensor in sensorTypes" :key="sensor" class="sensor-filter">
                  <input 
                    type="checkbox" 
                    :value="sensor" 
                    v-model="selectedSensors"
                    @change="updateSensorChart"
                  >
                  <span class="sensor-name">{{ getSensorDisplayName(sensor) }}</span>
                </label>
              </div>
            </div>
          </div>
          <div class="chart-container">
            <canvas ref="sensorTrendsChart" width="800" height="400"></canvas>
          </div>
        </div>

        <!-- График состояния систем -->
        <div class="chart-section">
          <div class="chart-header">
            <h3>🔧 Состояние систем</h3>
          </div>
          <div class="chart-container">
            <canvas ref="systemStatusChart" width="400" height="400"></canvas>
          </div>
        </div>

        <!-- График критических событий -->
        <div class="chart-section">
          <div class="chart-header">
            <h3>🚨 Критические события</h3>
          </div>
          <div class="chart-container">
            <canvas ref="criticalEventsChart" width="800" height="300"></canvas>
          </div>
        </div>

        <!-- Heatmap активности -->
        <div class="chart-section full-width">
          <div class="chart-header">
            <h3>🌡️ Карта активности систем</h3>
          </div>
          <div class="heatmap-container">
            <div class="heatmap-grid" ref="heatmapGrid">
              <!-- Динамически генерируется -->
            </div>
            <div class="heatmap-legend">
              <div class="legend-item">
                <div class="legend-color low"></div>
                <span>Низкая активность</span>
              </div>
              <div class="legend-item">
                <div class="legend-color medium"></div>
                <span>Средняя активность</span>
              </div>
              <div class="legend-item">
                <div class="legend-color high"></div>
                <span>Высокая активность</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Детальная аналитика по системам -->
      <div class="detailed-analytics" v-if="selectedSystem">
        <h2 class="section-title">🔍 Детальная аналитика</h2>
        
        <div class="analysis-tabs">
          <button 
            v-for="tab in analysisTabs" 
            :key="tab.id"
            class="tab-button"
            :class="{ active: activeAnalysisTab === tab.id }"
            @click="activeAnalysisTab = tab.id"
          >
            {{ tab.icon }} {{ tab.label }}
          </button>
        </div>

        <div class="tab-content">
          <!-- Вкладка производительности -->
          <div v-if="activeAnalysisTab === 'performance'" class="performance-analysis">
            <div class="performance-metrics">
              <div class="performance-card">
                <h4>📊 Средние значения</h4>
                <div class="performance-values">
                  <div 
                    v-for="(value, sensor) in averageValues" 
                    :key="sensor"
                    class="value-item"
                  >
                    <span class="sensor-name">{{ getSensorDisplayName(sensor) }}:</span>
                    <span class="sensor-value">{{ value.toFixed(2) }} {{ getSensorUnit(sensor) }}</span>
                  </div>
                </div>
              </div>
              
              <div class="performance-card">
                <h4>📈 Тренды</h4>
                <div class="trend-analysis">
                  <div 
                    v-for="trend in performanceTrends" 
                    :key="trend.sensor"
                    class="trend-item"
                  >
                    <span class="trend-sensor">{{ getSensorDisplayName(trend.sensor) }}:</span>
                    <span 
                      class="trend-value" 
                      :class="trend.direction"
                    >
                      {{ trend.direction === 'up' ? '📈' : trend.direction === 'down' ? '📉' : '➡️' }}
                      {{ Math.abs(trend.change).toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Вкладка диагностики -->
          <div v-if="activeAnalysisTab === 'diagnostics'" class="diagnostics-analysis">
            <div class="diagnostics-summary">
              <div class="diagnostic-card">
                <h4>🔍 Последняя диагностика</h4>
                <div v-if="lastDiagnostic" class="diagnostic-result">
                  <div class="diagnostic-health">
                    Здоровье системы: {{ lastDiagnostic.health_score }}%
                  </div>
                  <div class="diagnostic-status" :class="lastDiagnostic.status">
                    {{ getStatusText(lastDiagnostic.status) }}
                  </div>
                  <div class="diagnostic-recommendations">
                    <h5>Рекомендации:</h5>
                    <ul>
                      <li v-for="rec in lastDiagnostic.recommendations?.slice(0, 3)" :key="rec.id">
                        {{ rec.title }}
                      </li>
                    </ul>
                  </div>
                </div>
                <div v-else class="no-diagnostics">
                  <p>Диагностика еще не проводилась</p>
                  <button class="run-diagnostic-btn" @click="runDiagnostic">
                    🔍 Запустить диагностику
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Вкладка прогнозов -->
          <div v-if="activeAnalysisTab === 'predictions'" class="predictions-analysis">
            <div class="predictions-content">
              <div class="prediction-card">
                <h4>🔮 AI Прогнозы</h4>
                <div v-if="aiPredictions.length > 0" class="predictions-list">
                  <div 
                    v-for="prediction in aiPredictions" 
                    :key="prediction.id"
                    class="prediction-item"
                    :class="`priority-${prediction.priority}`"
                  >
                    <div class="prediction-header">
                      <span class="prediction-type">{{ prediction.component }}</span>
                      <span class="prediction-probability">{{ (prediction.probability * 100).toFixed(0) }}%</span>
                    </div>
                    <div class="prediction-description">{{ prediction.description }}</div>
                    <div class="prediction-timeline">
                      Ожидается: {{ prediction.time_to_failure }}
                    </div>
                  </div>
                </div>
                <div v-else class="no-predictions">
                  <p>Прогнозы недоступны</p>
                  <small>Недостаточно данных для AI анализа</small>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { Chart, registerables } from 'chart.js'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'

Chart.register(...registerables)

export default {
  name: 'Analytics',
  setup() {
    const router = useRouter()
    
    // Реактивные данные
    const isLoading = ref(false)
    const analyticsData = ref({})
    const availableSystems = ref([])
    const selectedSystem = ref('')
    const selectedPeriod = ref('30')
    const selectedSensors = ref(['pressure', 'temperature', 'flow'])
    const activeAnalysisTab = ref('performance')
    
    const sensorTypes = ['pressure', 'temperature', 'flow', 'vibration']
    const analysisTabs = [
      { id: 'performance', label: 'Производительность', icon: '📊' },
      { id: 'diagnostics', label: 'Диагностика', icon: '🔍' },
      { id: 'predictions', label: 'Прогнозы', icon: '🔮' }
    ]
    
    // Chart инстансы
    const sensorTrendsChart = ref(null)
    const systemStatusChart = ref(null)
    const criticalEventsChart = ref(null)
    const heatmapGrid = ref(null)
    
    const chartInstances = ref({})
    
    // Данные для детального анализа
    const averageValues = ref({})
    const performanceTrends = ref([])
    const lastDiagnostic = ref(null)
    const aiPredictions = ref([])
    
    // Вычисляемые свойства
    const averageHealth = computed(() => {
      if (!analyticsData.value.sensor_analytics) return 0
      
      // Простой расчет здоровья на основе критических событий
      const totalData = analyticsData.value.total_data_points || 0
      const criticalEvents = totalCriticalEvents.value
      
      if (totalData === 0) return 100
      const healthRatio = Math.max(0, 1 - (criticalEvents / totalData * 10))
      return Math.round(healthRatio * 100)
    })
    
    const totalCriticalEvents = computed(() => {
      if (!analyticsData.value.daily_trends) return 0
      return analyticsData.value.daily_trends.reduce((sum, day) => sum + (day.critical_events || 0), 0)
    })
    
    const averageUptime = computed(() => {
      if (!analyticsData.value.daily_trends) return 0
      
      const validDays = analyticsData.value.daily_trends.filter(day => day.total_readings > 0)
      if (validDays.length === 0) return 0
      
      return Math.round((validDays.length / analyticsData.value.daily_trends.length) * 100)
    })
    
    // Методы
    const loadAvailableSystems = async () => {
      try {
        const response = await hydraulicSystemService.getSystems()
        const systems = Array.isArray(response) ? response : response.results || []
        availableSystems.value = systems
      } catch (error) {
        console.error('Ошибка загрузки систем:', error)
      }
    }
    
    const loadAnalytics = async () => {
      if (!selectedSystem.value && availableSystems.value.length === 0) return
      
      try {
        isLoading.value = true
        
        if (selectedSystem.value) {
          // Загрузка аналитики для конкретной системы
          const response = await hydraulicSystemService.getSystemAnalytics(
            selectedSystem.value, 
            parseInt(selectedPeriod.value)
          )
          analyticsData.value = response
          
          // Загрузка дополнительных данных для детального анализа
          await loadDetailedAnalysis()
        } else {
          // Загрузка общей аналитики
          const allSystemsData = await Promise.all(
            availableSystems.value.slice(0, 5).map(system => 
              hydraulicSystemService.getSystemAnalytics(system.id, parseInt(selectedPeriod.value))
                .catch(error => {
                  console.error(`Ошибка загрузки данных системы ${system.id}:`, error)
                  return null
                })
            )
          )
          
          // Агрегация данных всех систем
          analyticsData.value = aggregateSystemsData(allSystemsData.filter(data => data !== null))
        }
        
        // Обновление графиков после загрузки данных
        await nextTick()
        updateCharts()
        
      } catch (error) {
        console.error('Ошибка загрузки аналитики:', error)
      } finally {
        isLoading.value = false
      }
    }
    
    const loadDetailedAnalysis = async () => {
      if (!selectedSystem.value) return
      
      try {
        // Расчет средних значений
        if (analyticsData.value.sensor_analytics) {
          const sensorData = analyticsData.value.sensor_analytics
          averageValues.value = {}
          
          Object.keys(sensorData).forEach(sensor => {
            if (sensorData[sensor].average) {
              averageValues.value[sensor] = sensorData[sensor].average
            }
          })
        }
        
        // Расчет трендов
        calculatePerformanceTrends()
        
        // Загрузка последней диагностики (имитация)
        // В реальном приложении это будет API вызов
        lastDiagnostic.value = {
          health_score: averageHealth.value,
          status: averageHealth.value >= 80 ? 'good' : averageHealth.value >= 60 ? 'fair' : 'poor',
          recommendations: [
            { id: 1, title: 'Проверить фильтры системы' },
            { id: 2, title: 'Контролировать температуру масла' },
            { id: 3, title: 'Оптимизировать рабочие циклы' }
          ]
        }
        
        // Генерация AI прогнозов (имитация)
        generateAIPredictions()
        
      } catch (error) {
        console.error('Ошибка загрузки детального анализа:', error)
      }
    }
    
    const aggregateSystemsData = (systemsData) => {
      // Простая агрегация данных нескольких систем
      const aggregated = {
        total_data_points: 0,
        sensor_analytics: {},
        daily_trends: [],
        generated_at: new Date().toISOString()
      }
      
      systemsData.forEach(data => {
        aggregated.total_data_points += data.total_data_points || 0
        
        // Агрегация данных датчиков
        Object.keys(data.sensor_analytics || {}).forEach(sensor => {
          if (!aggregated.sensor_analytics[sensor]) {
            aggregated.sensor_analytics[sensor] = {
              average: 0,
              minimum: Infinity,
              maximum: -Infinity,
              total_readings: 0,
              critical_events: 0
            }
          }
          
          const sensorData = data.sensor_analytics[sensor]
          const aggSensor = aggregated.sensor_analytics[sensor]
          
          aggSensor.average = (aggSensor.average * aggSensor.total_readings + 
                              sensorData.average * sensorData.total_readings) / 
                             (aggSensor.total_readings + sensorData.total_readings)
          aggSensor.minimum = Math.min(aggSensor.minimum, sensorData.minimum)
          aggSensor.maximum = Math.max(aggSensor.maximum, sensorData.maximum)
          aggSensor.total_readings += sensorData.total_readings
          aggSensor.critical_events += sensorData.critical_events
        })
      })
      
      return aggregated
    }
    
    const calculatePerformanceTrends = () => {
      // Расчет трендов производительности
      performanceTrends.value = []
      
      if (!analyticsData.value.sensor_analytics) return
      
      Object.keys(analyticsData.value.sensor_analytics).forEach(sensor => {
        const sensorData = analyticsData.value.sensor_analytics[sensor]
        
        // Простое определение тренда на основе данных
        const avgValue = sensorData.average || 0
        const maxValue = sensorData.maximum || 0
        const minValue = sensorData.minimum || 0
        
        let direction = 'stable'
        let change = 0
        
        // Простая логика определения тренда
        const variance = maxValue - minValue
        const normalizedVariance = avgValue > 0 ? (variance / avgValue) * 100 : 0
        
        if (normalizedVariance > 20) {
          direction = maxValue > avgValue * 1.1 ? 'up' : 'down'
          change = normalizedVariance
        }
        
        performanceTrends.value.push({
          sensor,
          direction,
          change
        })
      })
    }
    
    const generateAIPredictions = () => {
      // Имитация AI прогнозов на основе данных
      aiPredictions.value = []
      
      if (!analyticsData.value.sensor_analytics) return
      
      Object.keys(analyticsData.value.sensor_analytics).forEach(sensor => {
        const sensorData = analyticsData.value.sensor_analytics[sensor]
        
        // Генерация прогноза на основе критических событий
        const criticalRatio = sensorData.total_readings > 0 ? 
          sensorData.critical_events / sensorData.total_readings : 0
        
        if (criticalRatio > 0.1) { // Если больше 10% критических событий
          aiPredictions.value.push({
            id: aiPredictions.value.length + 1,
            component: getSensorDisplayName(sensor),
            probability: Math.min(criticalRatio * 2, 0.9),
            description: `Повышенный износ компонентов ${getSensorDisplayName(sensor)}`,
            time_to_failure: criticalRatio > 0.3 ? '1-2 недели' : '1-2 месяца',
            priority: criticalRatio > 0.3 ? 'high' : 'medium'
          })
        }
      })
      
      // Если нет прогнозов на основе данных, добавим общий
      if (aiPredictions.value.length === 0 && averageHealth.value < 80) {
        aiPredictions.value.push({
          id: 1,
          component: 'Общая система',
          probability: (100 - averageHealth.value) / 100,
          description: 'Рекомендуется профилактическое обслуживание',
          time_to_failure: '1-3 месяца',
          priority: 'medium'
        })
      }
    }
    
    const updateCharts = () => {
      createSensorTrendsChart()
      createSystemStatusChart()
      createCriticalEventsChart()
      createActivityHeatmap()
    }
    
    const createSensorTrendsChart = () => {
      if (!sensorTrendsChart.value || !analyticsData.value.daily_trends) return
      
      const ctx = sensorTrendsChart.value.getContext('2d')
      
      // Уничтожаем существующий график
      if (chartInstances.value.sensorTrends) {
        chartInstances.value.sensorTrends.destroy()
      }
      
      const labels = analyticsData.value.daily_trends.map(day => 
        new Date(day.date).toLocaleDateString('ru-RU', { month: 'short', day: 'numeric' })
      )
      
      const datasets = selectedSensors.value.map(sensor => {
        const color = getSensorColor(sensor)
        const data = analyticsData.value.daily_trends.map(day => {
          const sensorKey = `avg_${sensor}`
          return day[sensorKey] || 0
        })
        
        return {
          label: getSensorDisplayName(sensor),
          data: data,
          borderColor: color,
          backgroundColor: color + '20',
          fill: false,
          tension: 0.4
        }
      })
      
      chartInstances.value.sensorTrends = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'top'
            }
          },
          scales: {
            y: {
              beginAtZero: false,
              title: {
                display: true,
                text: 'Значения датчиков'
              }
            },
            x: {
              title: {
                display: true,
                text: 'Дата'
              }
            }
          }
        }
      })
    }
    
    const createSystemStatusChart = () => {
      if (!systemStatusChart.value) return
      
      const ctx = systemStatusChart.value.getContext('2d')
      
      if (chartInstances.value.systemStatus) {
        chartInstances.value.systemStatus.destroy()
      }
      
      // Данные для круговой диаграммы состояний
      const statusData = selectedSystem.value ? 
        [1, 0, 0] : // Для одной системы показываем активную
        [
          availableSystems.value.filter(s => s.status === 'active').length,
          availableSystems.value.filter(s => s.status === 'maintenance').length,
          availableSystems.value.filter(s => s.status === 'inactive').length
        ]
      
      chartInstances.value.systemStatus = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: ['Активные', 'Обслуживание', 'Неактивные'],
          datasets: [{
            data: statusData,
            backgroundColor: ['#48bb78', '#ed8936', '#a0aec0'],
            borderWidth: 2,
            borderColor: '#ffffff'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          }
        }
      })
    }
    
    const createCriticalEventsChart = () => {
      if (!criticalEventsChart.value || !analyticsData.value.daily_trends) return
      
      const ctx = criticalEventsChart.value.getContext('2d')
      
      if (chartInstances.value.criticalEvents) {
        chartInstances.value.criticalEvents.destroy()
      }
      
      const labels = analyticsData.value.daily_trends.map(day => 
        new Date(day.date).toLocaleDateString('ru-RU', { month: 'short', day: 'numeric' })
      )
      
      const data = analyticsData.value.daily_trends.map(day => day.critical_events || 0)
      
      chartInstances.value.criticalEvents = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Критические события',
            data: data,
            backgroundColor: '#f56565',
            borderColor: '#e53e3e',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Количество событий'
              }
            },
            x: {
              title: {
                display: true,
                text: 'Дата'
              }
            }
          }
        }
      })
    }
    
    const createActivityHeatmap = () => {
      if (!heatmapGrid.value || !analyticsData.value.daily_trends) return
      
      const grid = heatmapGrid.value
      grid.innerHTML = ''
      
      // Создаем сетку 7x24 (дни недели x часы)
      const days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
      const hours = Array.from({length: 24}, (_, i) => i)
      
      // Заголовки часов
      const hoursHeader = document.createElement('div')
      hoursHeader.className = 'heatmap-row'
      hoursHeader.innerHTML = '<div class="heatmap-cell header"></div>' + 
        hours.map(h => `<div class="heatmap-cell header">${h}</div>`).join('')
      grid.appendChild(hoursHeader)
      
      // Строки дней
      days.forEach((day, dayIndex) => {
        const dayRow = document.createElement('div')
        dayRow.className = 'heatmap-row'
        dayRow.innerHTML = `<div class="heatmap-cell header">${day}</div>`
        
        hours.forEach(hour => {
          const cell = document.createElement('div')
          cell.className = 'heatmap-cell'
          
          // Имитация данных активности
          const activity = Math.random() * 100
          let activityClass = 'low'
          if (activity > 70) activityClass = 'high'
          else if (activity > 30) activityClass = 'medium'
          
          cell.classList.add(activityClass)
          cell.title = `${day} ${hour}:00 - Активность: ${Math.round(activity)}%`
          
          dayRow.appendChild(cell)
        })
        
        grid.appendChild(dayRow)
      })
    }
    
    const updateSensorChart = () => {
      createSensorTrendsChart()
    }
    
    const runDiagnostic = async () => {
      if (!selectedSystem.value) return
      
      try {
        await hydraulicSystemService.runDiagnosis(selectedSystem.value)
        // Перезагружаем данные после диагностики
        await loadDetailedAnalysis()
      } catch (error) {
        console.error('Ошибка запуска диагностики:', error)
      }
    }
    
    const exportData = () => {
      // Простая реализация экспорта данных
      const dataToExport = {
        system: selectedSystem.value ? availableSystems.value.find(s => s.id == selectedSystem.value)?.name : 'Все системы',
        period: `${selectedPeriod.value} дней`,
        analytics: analyticsData.value,
        exportedAt: new Date().toISOString()
      }
      
      const dataStr = JSON.stringify(dataToExport, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      
      const link = document.createElement('a')
      link.href = URL.createObjectURL(dataBlob)
      link.download = `analytics_${selectedSystem.value || 'all'}_${new Date().toISOString().split('T')}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    
    // Вспомогательные функции
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
    
    const getSensorColor = (sensor) => {
      const colors = {
        'pressure': '#4299e1',
        'temperature': '#f56565',
        'flow': '#48bb78',
        'vibration': '#ed8936'
      }
      return colors[sensor] || '#a0aec0'
    }
    
    const getStatusText = (status) => {
      const texts = {
        'excellent': 'Отлично',
        'good': 'Хорошо',
        'fair': 'Удовлетворительно',
        'poor': 'Плохо',
        'critical': 'Критично'
      }
      return texts[status] || status
    }
    
    // Наблюдатели
    watch([selectedSystem, selectedPeriod], () => {
      if (availableSystems.value.length > 0) {
        loadAnalytics()
      }
    })
    
    // Жизненный цикл
    onMounted(async () => {
      await loadAvailableSystems()
      await loadAnalytics()
    })
    
    return {
      isLoading,
      analyticsData,
      availableSystems,
      selectedSystem,
      selectedPeriod,
      selectedSensors,
      activeAnalysisTab,
      sensorTypes,
      analysisTabs,
      sensorTrendsChart,
      systemStatusChart,
      criticalEventsChart,
      heatmapGrid,
      averageValues,
      performanceTrends,
      lastDiagnostic,
      aiPredictions,
      averageHealth,
      totalCriticalEvents,
      averageUptime,
      loadAnalytics,
      updateSensorChart,
      runDiagnostic,
      exportData,
      getSensorDisplayName,
      getSensorUnit,
      getStatusText
    }
  }
}
</script>

<style scoped>
/* Основные стили аналитики */
.analytics-container {
  padding: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

/* Заголовок аналитики */
.analytics-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 2rem;
  border-radius: 0 0 20px 20px;
  margin-bottom: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1.5rem;
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

.analytics-subtitle {
  color: #718096;
  font-size: 1.125rem;
  margin: 0;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  font-weight: 600;
  color: #4a5568;
  font-size: 0.875rem;
}

.system-selector,
.period-selector {
  padding: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: white;
  color: #4a5568;
  min-width: 150px;
}

.export-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.export-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* Загрузка */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  color: white;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

/* Основной контент */
.analytics-content {
  padding: 0 2rem 2rem;
}

/* Ключевые метрики */
.key-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
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
  transition: all 0.3s;
}

.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

.metric-icon {
  font-size: 2.5rem;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(102, 126, 234, 0.1);
}

.metric-data {
  flex: 1;
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
}

/* Сетка графиков */
.charts-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;
  margin-bottom: 3rem;
}

.chart-section {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.chart-section.full-width {
  grid-column: 1 / -1;
}

.chart-header {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  background: rgba(102, 126, 234, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-header h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #2d3748;
  margin: 0;
}

.chart-controls {
  display: flex;
  gap: 1rem;
}

.sensor-filters {
  display: flex;
  gap: 1rem;
}

.sensor-filter {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
}

.sensor-filter input[type="checkbox"] {
  margin: 0;
}

.chart-container {
  padding: 2rem;
  height: 400px;
}

/* Heatmap стили */
.heatmap-container {
  padding: 2rem;
}

.heatmap-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 2rem;
}

.heatmap-row {
  display: flex;
  gap: 2px;
}

.heatmap-cell {
  width: 20px;
  height: 20px;
  border-radius: 2px;
  cursor: pointer;
  transition: all 0.2s;
}

.heatmap-cell.header {
  background: transparent;
  font-size: 0.75rem;
  color: #718096;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  width: 30px;
}

.heatmap-cell.low { background: #e6fffa; }
.heatmap-cell.medium { background: #81e6d9; }
.heatmap-cell.high { background: #4fd1c7; }

.heatmap-cell:hover {
  transform: scale(1.2);
  z-index: 10;
  position: relative;
}

.heatmap-legend {
  display: flex;
  justify-content: center;
  gap: 2rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 2px;
}

/* Детальная аналитика */
.detailed-analytics {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #2d3748;
  margin: 0 0 2rem 0;
  padding: 2rem 2rem 0;
}

.analysis-tabs {
  display: flex;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  background: rgba(102, 126, 234, 0.05);
}

.tab-button {
  background: none;
  border: none;
  padding: 1rem 2rem;
  cursor: pointer;
  transition: all 0.2s;
  color: #718096;
  font-weight: 500;
}

.tab-button.active {
  color: #667eea;
  border-bottom: 3px solid #667eea;
  background: rgba(102, 126, 234, 0.1);
}

.tab-content {
  padding: 2rem;
}

/* Анализ производительности */
.performance-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.performance-card {
  background: #f7fafc;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.performance-card h4 {
  margin: 0 0 1rem 0;
  color: #2d3748;
  font-weight: 600;
}

.performance-values {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.value-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.sensor-name {
  color: #4a5568;
  font-weight: 500;
}

.sensor-value {
  color: #2d3748;
  font-weight: 600;
}

.trend-analysis {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.trend-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.trend-sensor {
  color: #4a5568;
  font-weight: 500;
}

.trend-value {
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.trend-value.up { color: #e53e3e; }
.trend-value.down { color: #38a169; }
.trend-value.stable { color: #4a5568; }

/* Диагностическая аналитика */
.diagnostics-summary {
  max-width: 600px;
}

.diagnostic-card {
  background: #f7fafc;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.diagnostic-card h4 {
  margin: 0 0 1rem 0;
  color: #2d3748;
  font-weight: 600;
}

.diagnostic-result {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.diagnostic-health {
  font-size: 1.125rem;
  font-weight: 600;
  color: #2d3748;
}

.diagnostic-status {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: 600;
  display: inline-block;
  width: fit-content;
}

.diagnostic-status.excellent { background: #c6f6d5; color: #22543d; }
.diagnostic-status.good { background: #bee3f8; color: #2a69ac; }
.diagnostic-status.fair { background: #fef5e7; color: #c05621; }
.diagnostic-status.poor { background: #fed7d7; color: #c53030; }

.diagnostic-recommendations h5 {
  margin: 0 0 0.5rem 0;
  color: #2d3748;
}

.diagnostic-recommendations ul {
  margin: 0;
  padding-left: 1.5rem;
  color: #4a5568;
}

.no-diagnostics {
  text-align: center;
  padding: 2rem;
  color: #718096;
}

.run-diagnostic-btn {
  background: #667eea;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 1rem;
}

/* Прогнозы */
.predictions-content {
  max-width: 800px;
}

.prediction-card {
  background: #f7fafc;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.prediction-card h4 {
  margin: 0 0 1rem 0;
  color: #2d3748;
  font-weight: 600;
}

.predictions-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.prediction-item {
  background: white;
  border-radius: 8px;
  padding: 1rem;
  border-left: 4px solid;
  transition: all 0.2s;
}

.prediction-item:hover {
  transform: translateX(4px);
}

.priority-high { border-left-color: #f56565; }
.priority-medium { border-left-color: #ed8936; }
.priority-low { border-left-color: #48bb78; }

.prediction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.prediction-type {
  font-weight: 600;
  color: #2d3748;
}

.prediction-probability {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.prediction-description {
  color: #4a5568;
  margin-bottom: 0.5rem;
}

.prediction-timeline {
  color: #718096;
  font-size: 0.875rem;
}

.no-predictions {
  text-align: center;
  padding: 2rem;
  color: #718096;
}

/* Адаптивность */
@media (max-width: 1200px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .performance-metrics {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .analytics-header {
    flex-direction: column;
    align-items: stretch;
    text-align: center;
  }
  
  .header-controls {
    justify-content: center;
  }
  
  .key-metrics {
    grid-template-columns: 1fr;
    padding: 0 1rem;
  }
  
  .analytics-content {
    padding: 0 1rem 1rem;
  }
  
  .metric-card {
    padding: 1.5rem;
  }
  
  .analysis-tabs {
    flex-direction: column;
  }
  
  .tab-button {
    text-align: center;
  }
}

@media (max-width: 480px) {
  .analytics-header {
    padding: 1.5rem;
  }
  
  .header-content h1 {
    font-size: 1.75rem;
  }
  
  .metric-icon {
    font-size: 2rem;
    width: 50px;
    height: 50px;
  }
  
  .metric-value {
    font-size: 2rem;
  }
  
  .heatmap-cell {
    width: 15px;
    height: 15px;
  }
  
  .heatmap-cell.header {
    width: 25px;
    font-size: 0.7rem;
  }
}
</style>
