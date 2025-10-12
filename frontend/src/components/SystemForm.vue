<template>
  <div class="modal-overlay" @click="$emit('close')">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>{{ isEdit ? '✏️ Редактировать систему' : '➕ Создать новую систему' }}</h3>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </div>

      <form @submit.prevent="submitForm" class="system-form">
        <div class="form-row">
          <div class="form-group">
            <label>Название системы *</label>
            <input
              v-model="form.name"
              type="text"
              :class="{ error: errors.name }"
              placeholder="Введите название системы"
              required
            />
            <span v-if="errors.name" class="error-message">{{ errors.name }}</span>
          </div>

          <div class="form-group">
            <label>Тип системы *</label>
            <select v-model="form.system_type" :class="{ error: errors.system_type }" required>
              <option value="">Выберите тип</option>
              <option value="industrial">Промышленная</option>
              <option value="mobile">Мобильная</option>
              <option value="marine">Морская</option>
              <option value="aviation">Авиационная</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Местоположение *</label>
            <input
              v-model="form.location"
              type="text"
              placeholder="Укажите местоположение"
              required
            />
          </div>

          <div class="form-group">
            <label>Статус</label>
            <select v-model="form.status">
              <option value="active">Активна</option>
              <option value="maintenance">На обслуживании</option>
              <option value="inactive">Неактивна</option>
              <option value="faulty">Неисправна</option>
            </select>
          </div>
        </div>

        <div class="form-section">
          <h4>🔧 Технические характеристики</h4>
          
          <div class="form-row">
            <div class="form-group">
              <label>Макс. давление (бар) *</label>
              <input
                v-model.number="form.max_pressure"
                type="number"
                step="0.1"
                min="0"
                placeholder="200.0"
                required
              />
            </div>

            <div class="form-group">
              <label>Расход (л/мин) *</label>
              <input
                v-model.number="form.flow_rate"
                type="number"
                step="0.1"
                min="0"
                placeholder="50.0"
                required
              />
            </div>
          </div>

          <div class="form-group">
            <label>Диапазон температур *</label>
            <input
              v-model="form.temperature_range"
              type="text"
              placeholder="-10°C до +80°C"
              required
            />
          </div>
        </div>

        <div class="form-actions">
          <button type="button" class="btn btn-secondary" @click="$emit('close')">
            Отмена
          </button>
          <button type="submit" class="btn" :disabled="loading">
            {{ loading ? '⏳ Сохранение...' : (isEdit ? '✅ Обновить' : '✨ Создать') }}
          </button>
        </div>

        <div v-if="submitError" class="error-banner">
          ❌ {{ submitError }}
        </div>
      </form>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'

export default {
  name: 'SystemForm',
  props: {
    system: {
      type: Object,
      default: null
    }
  },
  emits: ['close', 'success'],
  setup(props, { emit }) {
    const loading = ref(false)
    const submitError = ref('')
    const isEdit = ref(!!props.system)

    const form = reactive({
      name: '',
      system_type: '',
      location: '',
      status: 'active',
      max_pressure: null,
      flow_rate: null,
      temperature_range: ''
    })

    const errors = reactive({})

    // Заполнить форму данными для редактирования
    onMounted(() => {
      if (props.system) {
        Object.keys(form).forEach(key => {
          if (props.system[key] !== undefined) {
            form[key] = props.system[key]
          }
        })
      }
    })

    const validateForm = () => {
      Object.keys(errors).forEach(key => delete errors[key])

      if (!form.name?.trim()) {
        errors.name = 'Название обязательно'
      }
      if (!form.system_type) {
        errors.system_type = 'Выберите тип системы'
      }
      if (!form.max_pressure || form.max_pressure <= 0) {
        errors.max_pressure = 'Введите корректное давление'
      }
      if (!form.flow_rate || form.flow_rate <= 0) {
        errors.flow_rate = 'Введите корректный расход'
      }

      return Object.keys(errors).length === 0
    }

    const submitForm = async () => {
      if (!validateForm()) return

      loading.value = true
      submitError.value = ''

      try {
        if (isEdit.value) {
          await hydraulicSystemService.updateSystem(props.system.id, form)
        } else {
          await hydraulicSystemService.createSystem(form)
        }
        
        emit('success')
        emit('close')
      } catch (error) {
        console.error('Ошибка сохранения:', error)
        submitError.value = error.response?.data?.detail || error.response?.data?.error || 'Ошибка сохранения системы'
      } finally {
        loading.value = false
      }
    }

    return {
      form,
      errors,
      loading,
      submitError,
      isEdit,
      submitForm
    }
  }
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  padding: 1rem;
}

.modal-content {
  background: white;
  border-radius: 12px;
  width: 100%;
  max-width: 700px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px 12px 0 0;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.25rem;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: white;
  opacity: 0.8;
}

.close-btn:hover {
  opacity: 1;
}

.system-form {
  padding: 2rem;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.form-group label {
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #374151;
}

.form-group input,
.form-group select {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-group input.error,
.form-group select.error {
  border-color: #dc2626;
  box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
}

.error-message {
  color: #dc2626;
  font-size: 0.8rem;
  margin-top: 0.25rem;
}

.form-section {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #f3f4f6;
}

.form-section h4 {
  margin: 0 0 1.5rem 0;
  color: #667eea;
  font-size: 1.1rem;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e5e7eb;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  background: #667eea;
  color: white;
}

.btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(107, 114, 128, 0.3);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.error-banner {
  background: #fef2f2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  font-size: 0.9rem;
  border: 1px solid #fecaca;
}

@media (max-width: 640px) {
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .modal-content {
    margin: 1rem;
    max-height: 95vh;
  }

  .system-form {
    padding: 1.5rem;
  }

  .form-actions {
    flex-direction: column;
  }
}
</style>
