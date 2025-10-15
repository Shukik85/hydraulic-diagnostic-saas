<template>
  <div class="diagnostics-page">
    <div class="page-header">
      <h1>🔬 Диагностика систем</h1>
      <p>Запуск диагностики и анализ результатов</p>
    </div>

    <div class="diagnostics-content">
      <div class="system-selection">
        <h2>Выберите систему для диагностики</h2>
        <div class="systems-grid">
          <div 
            v-for="system in systems" 
            :key="system.id"
            class="system-card"
            :class="{ selected: selectedSystem?.id === system.id }"
            @click="selectSystem(system)"
          >
            <div class="system-icon">⚙️</div>
            <h3>{{ system.name }}</h3>
            <p>{{ system.system_type_display }}</p>
            <div class="system-status" :class="system.status">
              {{ system.status_display }}
            </div>
          </div>
        </div>
      </div>

      <div v-if="selectedSystem" class="diagnostic-controls">
        <h2>Параметры диагностики</h2>
        <div class="controls-grid">
          <div class="control-group">
            <label>Тип анализа:</label>
            <select v-model="diagnosticType">
              <option value="basic">Базовый</option>
              <option value="advanced">Расширенный</option>
              <option value="full">Полный</option>
            </select>
          </div>
          
          <div class="control-group">
            <label>Период анализа:</label>
            <select v-model="analysisPeriod">
              <option value="1">1 час</option>
              <option value="24">24 часа</option>
              <option value="168">7 дней</option>
              <option value="720">30 дней</option>
            </select>
          </div>
        </div>

        <div class="diagnostic-actions">
          <button 
            @click="runDiagnostic" 
            :disabled="isRunning"
            class="btn btn-primary"
          >
            <span v-if="isRunning">🔄 Выполняется...</span>
            <span v-else>🚀 Запустить диагностику</span>
          </button>
        </div>
      </div>

      <div v-if="diagnosticResults" class="diagnostic-results">
        <h2>Результаты диагностики</h2>
        <div class="results-summary">
          <div class="summary-card">
            <div class="summary-value">{{ diagnosticResults.health_score }}%</div>
            <div class="summary-label">Индекс здоровья</div>
          </div>
          
          <div class="summary-card">
            <div class="summary-value">{{ diagnosticResults.anomalies_found }}</div>
            <div class="summary-label">Аномалий найдено</div>
          </div>
          
          <div class="summary-card">
            <div class="summary-value">{{ Math.round(diagnosticResults.failure_probability * 100) }}%</div>
            <div class="summary-label">Риск отказа</div>
          </div>
        </div>

        <div class="results-details">
          <div class="recommendations">
            <h3>Рекомендации:</h3>
            <ul>
              <li>Проверить уровень гидравлической жидкости</li>
              <li>Осмотреть соединения на предмет утечек</li>
              <li>Запланировать замену фильтров через 2 недели</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  name: 'Diagnostics',
  setup() {
    const systems = ref([])
    const selectedSystem = ref(null)
    const diagnosticType = ref('basic')
    const analysisPeriod = ref('24')
    const isRunning = ref(false)
    const diagnosticResults = ref(null)

    const loadSystems = async () => {
      // Имитация загрузки систем
      systems.value = [
        {
          id: 1,
          name: 'Система гидропривода #1',
          system_type_display: 'Промышленная',
          status: 'active',
          status_display: 'Активна'
        },
        {
          id: 2,
          name: 'Система гидропривода #2',
          system_type_display: 'Мобильная',
          status: 'maintenance',
          status_display: 'На обслуживании'
        }
      ]
    }

    const selectSystem = (system) => {
      selectedSystem.value = system
      diagnosticResults.value = null
    }

    const runDiagnostic = async () => {
      isRunning.value = true
      
      try {
        // Имитация запроса диагностики
        await new Promise(resolve => setTimeout(resolve, 3000))
        
        diagnosticResults.value = {
          health_score: 85,
          anomalies_found: 2,
          failure_probability: 0.15,
          report_id: 'report_123'
        }
      } catch (error) {
        console.error('Ошибка диагностики:', error)
      } finally {
        isRunning.value = false
      }
    }

    onMounted(() => {
      loadSystems()
    })

    return {
      systems,
      selectedSystem,
      diagnosticType,
      analysisPeriod,
      isRunning,
      diagnosticResults,
      selectSystem,
      runDiagnostic
    }
  }
}
</script>

<style scoped>
.diagnostics-page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 3rem;
}

.page-header h1 {
  font-size: 2.5rem;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.systems-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.system-card {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.system-card:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.system-card.selected {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.system-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.system-status {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  margin-top: 1rem;
}

.system-status.active {
  background: #dcfce7;
  color: #166534;
}

.system-status.maintenance {
  background: #fef3c7;
  color: #92400e;
}

.diagnostic-controls {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 1.5rem 0;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  font-weight: 600;
  color: #374151;
}

.control-group select {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  background: white;
}

.diagnostic-actions {
  text-align: center;
  margin-top: 2rem;
}

.btn {
  padding: 1rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1.125rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.diagnostic-results {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.results-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.summary-card {
  text-align: center;
  padding: 1.5rem;
  background: #f8fafc;
  border-radius: 12px;
}

.summary-value {
  font-size: 2.5rem;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.summary-label {
  color: #64748b;
  font-weight: 500;
}

.recommendations ul {
  list-style: none;
  padding: 0;
}

.recommendations li {
  padding: 0.75rem;
  background: #f0fdf4;
  border-left: 4px solid #22c55e;
  margin-bottom: 0.5rem;
  border-radius: 4px;
}
</style>
