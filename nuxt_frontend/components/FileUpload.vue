<template>
  <div class="file-upload">
    <h3>Загрузить файл для диагностики</h3>
    <form @submit.prevent="uploadFile">
      <div class="form-group">
        <label>Выберите файл:</label>
        <input
          type="file"
          @change="handleFileChange"
          accept=".csv,.txt"
          required
        />
      </div>

      <div class="form-group">
        <label>ID Системы:</label>
        <input
          v-model="systemId"
          type="number"
          placeholder="Введите ID системы"
          required
        />
      </div>

      <button type="submit" :disabled="loading || !selectedFile">
        {{ loading ? 'Загрузка...' : 'Загрузить' }}
      </button>
    </form>

    <div v-if="message" :class="['message', messageType]">
      {{ message }}
    </div>

    <div v-if="diagnosticResult" class="result">
      <h4>Результат диагностики:</h4>
      <pre>{{ JSON.stringify(diagnosticResult, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const selectedFile = ref(null)
const systemId = ref('')
const loading = ref(false)
const message = ref('')
const messageType = ref('')
const diagnosticResult = ref(null)

const handleFileChange = (event) => {
  const file = event.target.files[0]
  selectedFile.value = file
  message.value = ''
  diagnosticResult.value = null
}

const uploadFile = async () => {
  if (!selectedFile.value || !systemId.value) {
    message.value = 'Пожалуйста, выберите файл и укажите ID системы'
    messageType.value = 'error'
    return
  }

  loading.value = true
  message.value = ''
  diagnosticResult.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('system_id', systemId.value)

    const config = useRuntimeConfig()
    const response = await $fetch(`${config.public.apiBase}/api/diagnostic/upload`, {
      method: 'POST',
      body: formData
    })

    message.value = 'Файл успешно загружен и обработан!'
    messageType.value = 'success'
    diagnosticResult.value = response.data

    // Очистка формы
    selectedFile.value = null
    systemId.value = ''
    event.target.reset()
  } catch (error) {
    console.error('Ошибка загрузки файла:', error)
    message.value = error.data?.message || 'Ошибка при загрузке файла'
    messageType.value = 'error'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.file-upload {
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

input[type="file"],
input[type="number"] {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  width: 100%;
  padding: 10px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

button:hover:not(:disabled) {
  background-color: #45a049;
}

button:disabled {
  background-color: #ccc;
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

.result {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
  border: 1px solid #dee2e6;
}

.result h4 {
  margin-top: 0;
  color: #333;
}

pre {
  background-color: #fff;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
  border: 1px solid #ddd;
}
</style>
