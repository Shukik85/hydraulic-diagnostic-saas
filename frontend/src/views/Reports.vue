<template>
  <div class="reports-page">
    <div class="page-header">
      <h1>📊 Отчеты</h1>
      <p>Просмотр и управление отчетами диагностики</p>
      <button @click="generateReport" class="btn btn-primary">
        📋 Новый отчет
      </button>
    </div>

    <div class="reports-filters">
      <div class="filter-group">
        <label>Система:</label>
        <select v-model="filters.system">
          <option value="">Все системы</option>
          <option value="1">Система #1</option>
          <option value="2">Система #2</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Тип отчета:</label>
        <select v-model="filters.type">
          <option value="">Все типы</option>
          <option value="automated">Автоматический</option>
          <option value="manual">Ручной</option>
          <option value="scheduled">Плановый</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Период:</label>
        <select v-model="filters.period">
          <option value="7">Последние 7 дней</option>
          <option value="30">Последние 30 дней</option>
          <option value="90">Последние 3 месяца</option>
        </select>
      </div>
    </div>

    <div class="reports-list">
      <div v-if="reports.length === 0" class="no-reports">
        <div class="no-reports-icon">📋</div>
        <h3>Отчеты не найдены</h3>
        <p>Создайте новый отчет или измените фильтры поиска</p>
      </div>

      <div v-else class="reports-grid">
        <div 
          v-for="report in reports" 
          :key="report.id"
          class="report-card"
          @click="viewReport(report)"
        >
          <div class="report-header">
            <h3>{{ report.title }}</h3>
            <span class="report-date">{{ formatDate(report.created_at) }}</span>
          </div>

          <div class="report-info">
            <div class="info-item">
              <span class="label">Система:</span>
              <span class="value">{{ report.system_name }}</span>
            </div>
            <div class="info-item">
              <span class="label">Тип:</span>
              <span class="value">{{ report.report_type_display }}</span>
            </div>
            <div class="info-item">
              <span class="label">Статус:</span>
              <span class="status" :class="report.status">{{ report.status_display }}</span>
            </div>
          </div>

          <div class="report-severity" :class="report.severity">
            {{ report.severity_display }}
          </div>

          <div class="report-actions">
            <button @click.stop="downloadReport(report)" class="btn-icon" title="Скачать">
              ⬇️
            </button>
            <button @click.stop="shareReport(report)" class="btn-icon" title="Поделиться">
              🔗
            </button>
            <button @click.stop="deleteReport(report)" class="btn-icon delete" title="Удалить">
              🗑️
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'

export default {
  name: 'Reports',
  setup() {
    const filters = ref({
      system: '',
      type: '',
      period: '30'
    })

    const allReports = ref([
      {
        id: 1,
        title: 'Диагностика системы гидропривода #1',
        system_name: 'Система #1',
        report_type: 'automated',
        report_type_display: 'Автоматический',
        status: 'completed',
        status_display: 'Завершен',
        severity: 'info',
        severity_display: 'Информация',
        created_at: '2023-12-01T10:30:00Z'
      },
      {
        id: 2,
        title: 'Плановая проверка системы #2',
        system_name: 'Система #2',
        report_type: 'scheduled',
        report_type_display: 'Плановый',
        status: 'completed',
        status_display: 'Завершен',
        severity: 'warning',
        severity_display: 'Предупреждение',
        created_at: '2023-11-28T15:45:00Z'
      },
      {
        id: 3,
        title: 'Анализ аномалий в системе #1',
        system_name: 'Система #1',
        report_type: 'manual',
        report_type_display: 'Ручной',
        status: 'processing',
        status_display: 'Обработка',
        severity: 'critical',
        severity_display: 'Критично',
        created_at: '2023-12-02T09:15:00Z'
      }
    ])

    const reports = computed(() => {
      return allReports.value.filter(report => {
        if (filters.value.system && report.system !== filters.value.system) {
          return false
        }
        if (filters.value.type && report.report_type !== filters.value.type) {
          return false
        }
        // Здесь можно добавить фильтрацию по периоду
        return true
      })
    })

    const formatDate = (dateString) => {
      if (!dateString) return ''
      return new Date(dateString).toLocaleDateString('ru-RU', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }

    const generateReport = () => {
      console.log('Создание нового отчета')
      // Здесь будет логика создания отчета
    }

    const viewReport = (report) => {
      console.log('Просмотр отчета:', report.id)
      // Здесь будет переход к детальному просмотру отчета
    }

    const downloadReport = (report) => {
      console.log('Скачивание отчета:', report.id)
      // Здесь будет логика скачивания отчета
    }

    const shareReport = (report) => {
      console.log('Поделиться отчетом:', report.id)
      // Здесь будет логика создания ссылки для sharing
    }

    const deleteReport = (report) => {
      if (confirm(`Удалить отчет "${report.title}"?`)) {
        const index = allReports.value.findIndex(r => r.id === report.id)
        if (index > -1) {
          allReports.value.splice(index, 1)
        }
      }
    }

    onMounted(() => {
      // Загрузка отчетов при монтировании компонента
      console.log('Загрузка отчетов...')
    })

    return {
      filters,
      reports,
      formatDate,
      generateReport,
      viewReport,
      downloadReport,
      shareReport,
      deleteReport
    }
  }
}
</script>

<style scoped>
.reports-page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.page-header h1 {
  font-size: 2.5rem;
  color: #2d3748;
}

.page-header p {
  color: #64748b;
  margin: 0.5rem 0;
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

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.reports-filters {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
}

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.filter-group label {
  font-weight: 600;
  color: #374151;
}

.filter-group select {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  background: white;
}

.no-reports {
  text-align: center;
  padding: 4rem 2rem;
  color: #64748b;
}

.no-reports-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.reports-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 1.5rem;
}

.report-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  position: relative;
}

.report-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.report-header h3 {
  margin: 0;
  color: #2d3748;
  font-size: 1.125rem;
}

.report-date {
  font-size: 0.875rem;
  color: #64748b;
}

.report-info {
  margin-bottom: 1rem;
}

.info-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.info-item .label {
  color: #64748b;
  font-weight: 500;
}

.info-item .value {
  color: #374151;
}

.status {
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
}

.status.completed {
  background: #dcfce7;
  color: #166534;
}

.status.processing {
  background: #fef3c7;
  color: #92400e;
}

.status.failed {
  background: #fee2e2;
  color: #dc2626;
}

.report-severity {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.report-severity.info {
  background: #dbeafe;
  color: #1e40af;
}

.report-severity.warning {
  background: #fef3c7;
  color: #92400e;
}

.report-severity.critical {
  background: #fee2e2;
  color: #dc2626;
}

.report-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  position: absolute;
  top: 1rem;
  right: 1rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.report-card:hover .report-actions {
  opacity: 1;
}

.btn-icon {
  background: none;
  border: none;
  padding: 0.5rem;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-icon:hover {
  background: #f1f5f9;
}

.btn-icon.delete:hover {
  background: #fee2e2;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }
  
  .reports-filters {
    grid-template-columns: 1fr;
  }
  
  .reports-grid {
    grid-template-columns: 1fr;
  }
}
</style>
