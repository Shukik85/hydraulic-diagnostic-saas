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
            <h4>{{ report.title }}</h4>
            <span class="report-date">{{ formatDate(report.created_at) }}</span>
          </div>
          <p v-if="report.description" class="report-description">{{ report.description }}</p>
          <div class="report-meta">
            <span>ID: {{ report.id }}</span>
            <span>Система: {{ report.system_id }}</span>
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
    type: [Number, String],
    required: true
  }
})

const { reports, loading, error, fetchReports, createReport } = useReports()

const newReport = ref({
  title: '',
  description: ''
})

// Загрузка отчетов при изменении системы
watch(() => props.systemId, async (newSystemId) => {
  if (newSystemId) {
    await fetchReports(newSystemId)
  }
}, { immediate: true })

// Обработчик создания отчета
const handleCreateReport = async () => {
  if (!newReport.value.title.trim()) return
  
  const success = await createReport(props.systemId, {
    title: newReport.value.title,
    description: newReport.value.description
  })
  
  if (success) {
    // Очистка формы после успешного создания
    newReport.value = {
      title: '',
      description: ''
    }
    // Перезагрузка списка отчетов
    await fetchReports(props.systemId)
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
</script>

<style scoped>
.reports-list {
  padding: 20px;
}

.reports-list h2 {
  margin-bottom: 20px;
  color: #333;
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
  color: #555;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 600;
  color: #333;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.form-group textarea {
  resize: vertical;
  font-family: inherit;
}

button[type="submit"] {
  background: #007bff;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
}

button[type="submit"]:hover:not(:disabled) {
  background: #0056b3;
}

button[type="submit"]:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.reports-container {
  margin-top: 20px;
}

.loading,
.error,
.empty {
  padding: 20px;
  text-align: center;
  color: #666;
}

.error {
  color: #d32f2f;
  background: #ffebee;
  border-radius: 4px;
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
  padding: 15px;
  margin-bottom: 15px;
  transition: box-shadow 0.3s ease;
}

.report-item:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.report-header h4 {
  margin: 0;
  color: #333;
  font-size: 18px;
}

.report-date {
  color: #888;
  font-size: 12px;
}

.report-description {
  color: #666;
  margin: 10px 0;
  line-height: 1.5;
}

.report-meta {
  display: flex;
  gap: 15px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #f0f0f0;
  font-size: 12px;
  color: #888;
}
</style>
