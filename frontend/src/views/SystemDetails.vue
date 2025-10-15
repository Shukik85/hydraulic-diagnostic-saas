<template>
  <div class="system-details-page">
    <div v-if="loading" class="loading">
      ⏳ Загрузка данных системы...
    </div>

    <div v-else-if="system" class="system-content">
      <!-- Заголовок системы -->
      <div class="system-header">
        <div class="header-main">
          <h1>{{ system.name }}</h1>
          <div class="system-meta">
            <span class="system-type">{{ system.system_type_display }}</span>
            <span class="system-status" :class="system.status">
              {{ system.status_display }}
            </span>
          </div>
        </div>
        
        <div class="header-actions">
          <button @click="runDiagnostic" class="btn btn-primary">
            🔬 Диагностика
          </button>
          <button @click="editSystem" class="btn btn-secondary">
            ✏️ Редактировать
          </button>
        </div>
      </div>

      <!-- Основные метрики -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">{{ system.health_score }}%</div>
          <div class="metric-label">Индекс здоровья</div>
          <div class="metric-trend positive">+2%</div>
        </div>
        
        <div class="metric-card">
          <div class="metric-value">{{ system.uptime || 96.5 }}%</div>
          <div class="metric-label">Время работы</div>
          <div class="metric-trend positive">+0.5%</div>
        </div>
        
        <div class="metric-card">
          <div class="metric-value">{{ criticalAlerts }}</div>
          <div class="metric-label">Критичных событий</div>
          <div class="metric-trend negative">+1</div>
        </div>
        
        <div class="metric-card">
          <div class="metric-value">{{ formatDate(system.last_maintenance) }}</div>
          <div class="metric-label">Последнее ТО</div>
          <div class="metric-info">{{ getDaysAgo(system.last_maintenance) }} дн. назад</div>
        </div>
      </div>

      <!-- Информация о системе и датчики -->
      <div class="content-grid">
        <!-- Основная информация -->
        <div class="info-card">
          <h2>📋 Информация о системе</h2>
          <div class="info-grid">
            <div class="info-item">
              <span class="label">Производитель:</span>
              <span class="value">{{ system.manufacturer || 'Не указан' }}</span>
            </div>
            <div class="info-item">
              <span class="label">Модель:</span>
              <span class="value">{{ system.model || 'Не указана' }}</span>
            </div>
            <div class="info-item">
              <span class="label">Серийный номер:</span>
              <span class="value">{{ system.serial_number || 'Не указан' }}</span>
            </div>
            <div class="info-item">
              <span class="label">Местоположение:</span>
              <span class="value">{{ system.location || 'Не указано' }}</span>
            </div>
            <div class="info-item">
              <span class="label">Макс. давление:</span>
              <span class="value">{{ system.max_pressure }} бар</span>
            </div>
            <div class="info-item">
              <span class="label">Макс. расход:</span>
              <span class="value">{{ system.max_flow }} л/мин</span>
            </div>
          </div>
        </div>

        <!-- Текущие показания датчиков -->
        <div class="sensors-card">
          <h2>📊 Текущие показания</h2>
          <div v-if="sensorData.length === 0" class="no-data">
            Нет данных от датчиков
          </div>
          <div v-else class="sensors-list">
            <div 
              v-for="sensor in sensorData" 
              :key="sensor.type"
              class="sensor-item"
              :class="{ critical: sensor.is_critical }"
            >
              <div class="sensor-icon">{{ getSensorIcon(sensor.type) }}</div>
              <div class="sensor-info">
                <div class="sensor-name">{{ getSensorName(sensor.type) }}</div>
                <div class="sensor-value">
                  {{ sensor.value }} {{ sensor.unit }}
                </div>
              </div>
              <div v-if="sensor.is_critical" class="sensor-alert">⚠️</div>
            </div>
          </div>
        </div>

        <!-- График показаний (заглушка) -->
        <div class="chart-card">
          <h2>📈 Тренды показаний</h2>
          <div class="chart-placeholder">
            <div class="chart-mockup">
              📈 График показаний за последние 24 часа
              <br><small>(Интеграция с Chart.js в разработке)</small>
            </div>
          </div>
        </div>

        <!-- Недавние события -->
        <div class="events-card">
          <h2>🔔 Недавние события</h2>
          <div class="events-list">
            <div class="event-item warning">
              <div class="event-time">10:30</div>
              <div class="event-text">
                Превышение температуры: 85°C
              </div>
            </div>
            <div class="event-item info">
              <div class="event-time">09:15</div>
              <div class="event-text">
                Плановое обслуживание выполнено
              </div>
            </div>
            <div class="event-item normal">
              <div class="event-time">08:00</div>
              <div class="event-text">
                Система запущена
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="error">
      ❌ Система не найдена
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'

export default {
  name: 'SystemDetails',
  setup() {
    const route = useRoute()
    const loading = ref(true)
    const system = ref(null)
    const sensorData = ref([])

    const criticalAlerts = computed(() => {
      return sensorData.value.filter(s => s.is_critical).length
    })

    const loadSystem = async () => {
      const systemId = route.params.id
      
      try {
        // Имитация загрузки данных системы
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        system.value = {
          id: systemId,
          name: `Гидравлическая система #${systemId}`,
          system_type_display: 'Промышленная',
          status: 'active',
          status_display: 'Активна',
          health_score: 85,
          manufacturer: 'Bosch Rexroth',
          model: 'HYD-3000',
          serial_number: 'SN123456789',
          location: 'Цех №1, участок А',
          max_pressure: 350,
          max_flow: 120,
          last_maintenance: '2023-11-15T10:00:00Z'
        }

        sensorData.value = [
          {
            type: 'pressure',
            value: 285,
            unit: 'бар',
            is_critical: false
          },
          {
            type: 'temperature',
            value: 85,
            unit: '°C',
            is_critical: true
          },
          {
            type: 'flow',
            value: 95,
            unit: 'л/мин',
            is_critical: false
          },
          {
            type: 'vibration',
            value: 12,
            unit: 'мм/с',
            is_critical: false
          }
        ]
        
      } catch (error) {
        console.error('Ошибка загрузки системы:', error)
      } finally {
        loading.value = false
      }
    }

    const getSensorIcon = (type) => {
      const icons = {
        pressure: '💨',
        temperature: '🌡️',
        flow: '🌊',
        vibration: '📳'
      }
      return icons[type] || '📊'
    }

    const getSensorName = (type) => {
      const names = {
        pressure: 'Давление',
        temperature: 'Температура',
        flow: 'Расход',
        vibration: 'Вибрация'
      }
      return names[type] || 'Датчик'
    }

    const formatDate = (dateString) => {
      if (!dateString) return 'Не указано'
      return new Date(dateString).toLocaleDateString('ru-RU')
    }

    const getDaysAgo = (dateString) => {
      if (!dateString) return 0
      const date = new Date(dateString)
      const now = new Date()
      const diffTime = Math.abs(now - date)
      return Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    }

    const runDiagnostic = () => {
      console.log('Запуск диагностики системы:', system.value.id)
      // Здесь будет логика запуска диагностики
    }

    const editSystem = () => {
      console.log('Редактирование системы:', system.value.id)
      // Здесь будет переход к форме редактирования
    }

    onMounted(() => {
      loadSystem()
    })

    return {
      loading,
      system,
      sensorData,
      criticalAlerts,
      getSensorIcon,
      getSensorName,
      formatDate,
      getDaysAgo,
      runDiagnostic,
      editSystem
    }
  }
}
</script>

<style scoped>
.system-details-page {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.loading {
  text-align: center;
  padding: 4rem;
  font-size: 1.25rem;
  color: #64748b;
}

.system-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2rem;
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.header-main h1 {
  font-size: 2rem;
  color: #2d3748;
  margin-bottom: 1rem;
}

.system-meta {
  display: flex;
  gap: 1rem;
}

.system-type {
  background: #dbeafe;
  color: #1e40af;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.system-status {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
}

.system-status.active {
  background: #dcfce7;
  color: #166534;
}

.system-status.maintenance {
  background: #fef3c7;
  color: #92400e;
}

.header-actions {
  display: flex;
  gap: 1rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.btn-secondary {
  background: #f1f5f9;
  color: #64748b;
  border: 1px solid #e2e8f0;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.metric-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  text-align: center;
}

.metric-value {
  font-size: 2.5rem;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.metric-label {
  color: #64748b;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.metric-trend {
  font-size: 0.875rem;
  font-weight: 600;
}

.metric-trend.positive {
  color: #059669;
}

.metric-trend.negative {
  color: #dc2626;
}

.metric-info {
  font-size: 0.875rem;
  color: #64748b;
}

.content-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
}

.info-card,
.sensors-card,
.chart-card,
.events-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.info-card h2,
.sensors-card h2,
.chart-card h2,
.events-card h2 {
  margin-bottom: 1.5rem;
  color: #2d3748;
}

.info-grid {
  display: grid;
  gap: 1rem;
}

.info-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f1f5f9;
}

.info-item .label {
  color: #64748b;
  font-weight: 500;
}

.info-item .value {
  color: #374151;
  font-weight: 600;
}

.sensors-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.sensor-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
  transition: all 0.2s;
}

.sensor-item.critical {
  background: #fee2e2;
  border-left: 4px solid #dc2626;
}

.sensor-icon {
  font-size: 1.5rem;
}

.sensor-info {
  flex: 1;
}

.sensor-name {
  color: #64748b;
  font-size: 0.875rem;
  margin-bottom: 0.25rem;
}

.sensor-value {
  color: #2d3748;
  font-weight: 600;
  font-size: 1.125rem;
}

.sensor-alert {
  font-size: 1.25rem;
}

.chart-placeholder {
  height: 200px;
  background: #f8fafc;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chart-mockup {
  text-align: center;
  color: #64748b;
  font-size: 1.125rem;
}

.events-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.event-item {
  display: flex;
  gap: 1rem;
  padding: 0.75rem;
  border-radius: 8px;
}

.event-item.warning {
  background: #fef3c7;
  border-left: 4px solid #f59e0b;
}

.event-item.info {
  background: #dbeafe;
  border-left: 4px solid #3b82f6;
}

.event-item.normal {
  background: #f0fdf4;
  border-left: 4px solid #22c55e;
}

.event-time {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 600;
  min-width: 60px;
}

.event-text {
  color: #374151;
}

.no-data {
  text-align: center;
  color: #64748b;
  padding: 2rem;
  font-style: italic;
}

.error {
  text-align: center;
  padding: 4rem;
  font-size: 1.25rem;
  color: #dc2626;
}

@media (max-width: 768px) {
  .system-header {
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .header-actions {
    width: 100%;
    justify-content: center;
  }
  
  .content-grid {
    grid-template-columns: 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
