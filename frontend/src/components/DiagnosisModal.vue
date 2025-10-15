<template>
  <div class="diagnosis-modal-overlay" @click.self="$emit('close')">
    <div class="diagnosis-modal">
      <div class="modal-header">
        <h3>🔬 Результаты диагностики</h3>
        <button @click="$emit('close')" class="close-btn">×</button>
      </div>
      
      <div class="modal-body">
        <div class="diagnosis-results">
          <div class="health-score">
            <div class="score-circle">{{ healthScore }}%</div>
            <p>Индекс здоровья системы</p>
          </div>
          
          <div class="findings">
            <h4>Основные выводы:</h4>
            <ul>
              <li>Система функционирует в пределах нормы</li>
              <li>Обнаружено 2 незначительные аномалии</li>
              <li>Рекомендуется плановое обслуживание через 30 дней</li>
            </ul>
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <button @click="downloadReport" class="btn btn-primary">Скачать отчет</button>
        <button @click="$emit('close')" class="btn btn-secondary">Закрыть</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DiagnosisModal',
  props: {
    healthScore: {
      type: Number,
      default: 85
    }
  },
  emits: ['close'],
  methods: {
    downloadReport() {
      console.log('Скачивание отчета...')
      this.$emit('close')
    }
  }
}
</script>

<style scoped>
.diagnosis-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.diagnosis-modal {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.close-btn {
  background: none;
  border: none;
  font-size: 2rem;
  cursor: pointer;
  color: #64748b;
}

.modal-body {
  padding: 1.5rem;
}

.health-score {
  text-align: center;
  margin-bottom: 2rem;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  font-weight: bold;
  margin: 0 auto 1rem;
}

.findings ul {
  list-style: none;
  padding: 0;
}

.findings li {
  padding: 0.5rem 0;
  border-bottom: 1px solid #f1f5f9;
}

.modal-footer {
  display: flex;
  gap: 1rem;
  padding: 1.5rem;
  border-top: 1px solid #e2e8f0;
  justify-content: flex-end;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.btn-secondary {
  background: #f1f5f9;
  color: #64748b;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
</style>
