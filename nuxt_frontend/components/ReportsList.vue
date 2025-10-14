<template>
  <div class="reports-list">
    <h2>Отчеты системы</h2>
    
    <!-- Форма создания нового отчета -->
    <div class="report-form">
      <h3>Создать новый отчет</h3>
      <form @submit.prevent="handleCreateReport">
        <div class="form-group">
          <label for="reportTitle">Название отчета:</label>
          <input 
            id="reportTitle"
            v-model="newReport.title"
            type="text"
            required
            placeholder="Введите название отчета"
          />
        </div>
        
        <div class="form-group">
          <label for="reportDescription">Описание:</label>
          <textarea 
            id="reportDescription"
            v-model="newReport.description"
            rows="3"
            placeholder="Описание отчета"
          />
        </div>
        
        <button type="submit" :disabled="loading">{{ loading ? 'Создание...' : 'Создать отчет' }}</button>
      </form>
    </div>
    
    <!-- Список отчетов -->
    <div class="reports-container">
      <div v-if="loading && !reports.length" class="loading">Загрузка отчетов...</div>
      <div v-else-if="error" class="error">{{ error }}</div>
      <div v-else-if="!reports.length" class="empty">Нет отчетов для выбранной системы</div>
      
      <ul v-else class="reports-list-items">
        <li v-for="report in reports" :key="report.id" class="report-item">
          <div class="report-header">
            <div class="report-title">
              {{ report.title }}
              <span class="report-date">{{ formatDate(report.created_at) }}</span>
            </div>
            <div class="report-actions">
              <button 
                @click="toggleExportMenu(report.id)"
                class="export-button"
                :class="{ 'active': activeExportMenu === report.id }"
              >
                Экспорт
              </button>
              <div v-if="activeExportMenu === report.id" class="export-menu">
                <button @click="downloadReport(report.id, 'csv')" class="export-option">
                  CSV
                </button>
                <button @click="downloadReport(report.id, 'json')" class="export-option">
                  JSON
                </button>
              </div>
            </div>
          </div>
          <p v-if="report.description" class="report-description">{{ report.description }}</p>
          <div class="report-meta">
            ID: {{ report.id }}
            Система: {{ report.system_id }}
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useReports } from '~/composables/useReports'

const props = defineProps({
  systemId: {
    type: Number,
    required: true
  }
})

const { reports, loading, error, fetchReports, createReport } = useReports()

const newReport = ref({
  title: '',
  description: ''
})

const activeExportMenu = ref(null)

// Загрузка отчетов при изменении systemId
watch(() => props.systemId, async (newSystemId) => {
  if (newSystemId) {
    await fetchReports(newSystemId)
  }
}, { immediate: true })

// Создание нового отчета
const handleCreateReport = async () => {
  if (!props.systemId) {
    error.value = 'Не выбрана система'
    return
  }
  
  try {
    await createReport(props.systemId, newReport.value)
    newReport.value = { title: '', description: '' }
    await fetchReports(props.systemId)
  } catch (e) {
    console.error('Ошибка создания отчета:', e)
  }
}

// Форматирование даты
const formatDate = (dateString) => {
  if (!dateString) return ''
  const date = new Date(dateString)
  return date.toLocaleString('ru-RU', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Переключение меню экспорта
const toggleExportMenu = (reportId) => {
  if (activeExportMenu.value === reportId) {
    activeExportMenu.value = null
  } else {
    activeExportMenu.value = reportId
  }
}

// Экспорт отчета
const downloadReport = async (reportId, format) => {
  try {
    const config = useRuntimeConfig()
    const apiUrl = `${config.public.apiUrl}/systems/${props.systemId}/reports/${reportId}/export/?format=${format}`
    
    // Запрос к backend API
    const response = await fetch(apiUrl, {
      method: 'GET',
      headers: {
        'Accept': format === 'csv' ? 'text/csv' : 'application/json'
      }
    })
    
    if (!response.ok) {
      throw new Error(`Ошибка экспорта: ${response.statusText}`)
    }
    
    // Получение файла как blob
    const blob = await response.blob()
    
    // Создание ссылки для скачивания
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `report_${reportId}.${format}`
    
    // Инициирование скачивания
    document.body.appendChild(link)
    link.click()
    
    // Очистка
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
    // Закрытие меню экспорта
    activeExportMenu.value = null
  } catch (e) {
    console.error('Ошибка при экспорте отчета:', e)
    error.value = `Ошибка экспорта: ${e.message}`
  }
}
</script>

<style scoped>
.reports-list {
  padding: 20px;
}

.report-form {
  background: #f5f5f5;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.report-form h3 {
  margin-top: 0;
  margin-bottom: 15px;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-family: inherit;
}

.reports-container {
  margin-top: 20px;
}

.loading,
.error,
.empty {
  padding: 20px;
  text-align: center;
  border-radius: 4px;
}

.loading {
  background: #e3f2fd;
  color: #1976d2;
}

.error {
  background: #ffebee;
  color: #c62828;
}

.empty {
  background: #f5f5f5;
  color: #666;
}

.reports-list-items {
  list-style: none;
  padding: 0;
  margin: 0;
}

.report-item {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}

.report-item:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.report-title {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 18px;
  font-weight: 600;
  color: #333;
  flex: 1;
}

.report-date {
  font-size: 14px;
  color: #666;
  font-weight: normal;
}

.report-actions {
  position: relative;
}

.export-button {
  padding: 8px 16px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.export-button:hover {
  background: #45a049;
}

.export-button.active {
  background: #45a049;
}

.export-menu {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 5px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  z-index: 10;
  min-width: 120px;
}

.export-option {
  display: block;
  width: 100%;
  padding: 10px 15px;
  background: white;
  border: none;
  text-align: left;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.export-option:hover {
  background: #f5f5f5;
}

.export-option:first-child {
  border-radius: 4px 4px 0 0;
}

.export-option:last-child {
  border-radius: 0 0 4px 4px;
}

.report-description {
  color: #666;
  margin: 10px 0;
  line-height: 1.5;
}

.report-meta {
  font-size: 14px;
  color: #999;
  margin-top: 10px;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
