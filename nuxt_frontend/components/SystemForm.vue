<template>
  <div class="system-form">
    <h3>{{ isEditMode ? 'Редактировать систему' : 'Добавить систему' }}</h3>
    <form @submit.prevent="handleSubmit">
      <div class="form-group">
        <label for="name">Название системы:</label>
        <input
          id="name"
          v-model="formData.name"
          type="text"
          placeholder="Введите название"
          required
        />
        <span v-if="errors.name" class="error-message">{{ errors.name }}</span>
      </div>

      <div class="form-group">
        <label for="description">Описание:</label>
        <textarea
          id="description"
          v-model="formData.description"
          placeholder="Введите описание"
          rows="4"
        ></textarea>
        <span v-if="errors.description" class="error-message">{{ errors.description }}</span>
      </div>

      <div class="form-group">
        <label for="location">Расположение:</label>
        <input
          id="location"
          v-model="formData.location"
          type="text"
          placeholder="Введите расположение"
        />
        <span v-if="errors.location" class="error-message">{{ errors.location }}</span>
      </div>

      <div class="form-actions">
        <button type="submit" :disabled="loading">
          {{ loading ? 'Сохранение...' : 'Сохранить' }}
        </button>
        <button type="button" @click="handleCancel" :disabled="loading">
          Отмена
        </button>
      </div>
    </form>

    <div v-if="message" :class="['message', messageType]">
      {{ message }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  system: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['submit', 'cancel'])

const { createSystem, updateSystem } = useSystems()

const formData = ref({
  name: '',
  description: '',
  location: ''
})

const errors = ref({
  name: '',
  description: '',
  location: ''
})

const loading = ref(false)
const message = ref('')
const messageType = ref('')

const isEditMode = computed(() => !!props.system)

// Заполнение формы при редактировании
watch(
  () => props.system,
  (newSystem) => {
    if (newSystem) {
      formData.value = {
        name: newSystem.name || '',
        description: newSystem.description || '',
        location: newSystem.location || ''
      }
    } else {
      resetForm()
    }
  },
  { immediate: true }
)

const validateForm = () => {
  errors.value = {
    name: '',
    description: '',
    location: ''
  }

  let isValid = true

  if (!formData.value.name || formData.value.name.trim() === '') {
    errors.value.name = 'Название обязательно'
    isValid = false
  } else if (formData.value.name.length < 3) {
    errors.value.name = 'Название должно содержать минимум 3 символа'
    isValid = false
  }

  if (formData.value.description && formData.value.description.length > 500) {
    errors.value.description = 'Описание не должно превышать 500 символов'
    isValid = false
  }

  return isValid
}

const handleSubmit = async () => {
  if (!validateForm()) {
    return
  }

  loading.value = true
  message.value = ''

  try {
    if (isEditMode.value) {
      await updateSystem(props.system.id, formData.value)
      message.value = 'Система успешно обновлена!'
    } else {
      await createSystem(formData.value)
      message.value = 'Система успешно создана!'
      resetForm()
    }
    messageType.value = 'success'
    emit('submit', formData.value)
  } catch (error) {
    console.error('Ошибка сохранения системы:', error)
    message.value = error.data?.message || 'Ошибка при сохранении системы'
    messageType.value = 'error'
  } finally {
    loading.value = false
  }
}

const handleCancel = () => {
  resetForm()
  emit('cancel')
}

const resetForm = () => {
  formData.value = {
    name: '',
    description: '',
    location: ''
  }
  errors.value = {
    name: '',
    description: '',
    location: ''
  }
  message.value = ''
}
</script>

<style scoped>
.system-form {
  max-width: 600px;
  margin: 20px auto;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h3 {
  margin-top: 0;
  color: #333;
  margin-bottom: 20px;
}

.form-group {
  margin-bottom: 15px;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555;
}

input[type="text"],
textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-sizing: border-box;
  font-family: inherit;
}

textarea {
  resize: vertical;
}

.error-message {
  color: #d32f2f;
  font-size: 0.875rem;
  margin-top: 4px;
  display: block;
}

.form-actions {
  display: flex;
  gap: 10px;
  margin-top: 20px;
}

button {
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

button[type="submit"] {
  background-color: #4CAF50;
  color: white;
}

button[type="submit"]:hover:not(:disabled) {
  background-color: #45a049;
}

button[type="button"] {
  background-color: #f5f5f5;
  color: #333;
  border: 1px solid #ddd;
}

button[type="button"]:hover:not(:disabled) {
  background-color: #e0e0e0;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.message {
  margin-top: 15px;
  padding: 10px;
  border-radius: 4px;
  text-align: center;
}

.message.success {
  background-color: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}

.message.error {
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}
</style>
