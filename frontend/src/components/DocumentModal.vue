<template>
  <div class="document-modal-overlay" @click.self="$emit('close')">
    <div class="document-modal">
      <div class="modal-header">
        <h3>📄 {{ document?.title || 'Документ' }}</h3>
        <button @click="$emit('close')" class="close-btn">×</button>
      </div>
      
      <div class="modal-body">
        <div v-if="document" class="document-content">
          <div class="document-info">
            <p><strong>Категория:</strong> {{ document.category }}</p>
            <p><strong>Дата обновления:</strong> {{ formatDate(document.updated_at) }}</p>
          </div>
          
          <div class="document-text">
            {{ document.content }}
          </div>
        </div>
        
        <div v-else class="no-document">
          <p>Документ не найден</p>
        </div>
      </div>
      
      <div class="modal-footer">
        <button @click="copyContent" class="btn btn-primary">Копировать</button>
        <button @click="$emit('close')" class="btn btn-secondary">Закрыть</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DocumentModal',
  props: {
    document: {
      type: Object,
      default: null
    }
  },
  emits: ['close'],
  methods: {
    formatDate(dateString) {
      if (!dateString) return 'Не указано'
      return new Date(dateString).toLocaleDateString('ru-RU')
    },
    async copyContent() {
      if (this.document?.content) {
        try {
          await navigator.clipboard.writeText(this.document.content)
          console.log('Содержимое скопировано')
        } catch (error) {
          console.error('Ошибка копирования:', error)
        }
      }
    }
  }
}
</script>

<style scoped>
.document-modal-overlay {
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

.document-modal {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 700px;
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
  max-height: 60vh;
  overflow-y: auto;
}

.document-info {
  background: #f8fafc;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.document-text {
  line-height: 1.6;
  color: #374151;
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
