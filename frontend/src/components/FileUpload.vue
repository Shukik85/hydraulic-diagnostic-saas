<template>
  <div class="modal-overlay" @click="$emit('close')">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>📊 Загрузка данных датчиков</h3>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </div>

      <div class="upload-body">
        <div class="format-selector">
          <label>Формат файла:</label>
          <select v-model="fileFormat">
            <option value="csv">CSV файл</option>
            <option value="json">JSON файл</option>
          </select>
        </div>

        <div class="upload-zone" 
             :class="{ 'drag-over': dragOver, 'has-file': selectedFile }"
             @drop="handleDrop" 
             @dragover.prevent="dragOver = true"
             @dragleave="dragOver = false">
          
          <div v-if="!selectedFile" class="upload-placeholder">
            <div class="upload-icon">📁</div>
            <p>Перетащите файл сюда или</p>
            <button type="button" class="btn" @click="$refs.fileInput.click()">Выберите файл</button>
            <p class="file-info">Поддерживаются CSV и JSON файлы</p>
          </div>

          <div v-else class="file-selected">
            <div class="file-icon">📄</div>
            <div class="file-details">
              <p><strong>{{ selectedFile.name }}</strong></p>
              <p>{{ formatFileSize(selectedFile.size) }}</p>
            </div>
            <button type="button" class="btn btn-secondary" @click="removeFile">Удалить</button>
          </div>

          <input 
            ref="fileInput" 
            type="file" 
            :accept="fileFormat === 'csv' ? '.csv' : '.json'"
            @change="handleFileSelect" 
            style="display: none"
          />
        </div>

        <div class="format-help">
          <h4>Пример формата {{ fileFormat.toUpperCase() }}:</h4>
          <pre v-if="fileFormat === 'csv'" class="code-example">timestamp,sensor_type,value,unit,is_critical,warning_message
2025-10-12T15:30:00Z,pressure,150.5,bar,false,
2025-10-12T15:31:00Z,temperature,65.2,°C,false,</pre>
          
          <pre v-else class="code-example">[
  {
    "timestamp": "2025-10-12T15:30:00Z",
    "sensor_type": "pressure",
    "value": 150.5,
    "unit": "bar",
    "is_critical": false
  }
]</pre>
        </div>

        <div class="upload-actions">
          <button type="button" class="btn btn-secondary" @click="$emit('close')">Отмена</button>
          <button 
            type="button"
            class="btn" 
            :disabled="!selectedFile || uploading" 
            @click="uploadFile"
          >
            {{ uploading ? '⏳ Загрузка...' : '📤 Загрузить' }}
          </button>
        </div>

        <div v-if="uploadError" class="error-banner">
          ❌ {{ uploadError }}
        </div>

        <div v-if="uploadSuccess" class="success-banner">
          ✅ {{ uploadSuccess }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'

export default {
  name: 'FileUpload',
  props: {
    systemId: {
      type: Number,
      required: true
    }
  },
  emits: ['close', 'success'],
  setup(props, { emit }) {
    const selectedFile = ref(null)
    const fileFormat = ref('csv')
    const dragOver = ref(false)
    const uploading = ref(false)
    const uploadError = ref('')
    const uploadSuccess = ref('')

    const handleFileSelect = (event) => {
      const file = event.target.files
      if (file) {
        selectedFile.value = file
        uploadError.value = ''
        uploadSuccess.value = ''
      }
    }

    const handleDrop = (event) => {
      event.preventDefault()
      dragOver.value = false
      
      const files = event.dataTransfer.files
      if (files.length > 0) {
        selectedFile.value = files
        uploadError.value = ''
        uploadSuccess.value = ''
      }
    }

    const removeFile = () => {
      selectedFile.value = null
      uploadError.value = ''
      uploadSuccess.value = ''
    }

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }

    const uploadFile = async () => {
      if (!selectedFile.value) return

      uploading.value = true
      uploadError.value = ''
      uploadSuccess.value = ''

      try {
        const formData = new FormData()
        formData.append('file', selectedFile.value)
        formData.append('format', fileFormat.value)

        // Пока что просто симулируем успешную загрузку
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        uploadSuccess.value = `Файл "${selectedFile.value.name}" успешно загружен!`
        
        setTimeout(() => {
          emit('success')
          emit('close')
        }, 1500)

      } catch (error) {
        console.error('Upload error:', error)
        uploadError.value = error.response?.data?.error || 'Ошибка загрузки файла'
      } finally {
        uploading.value = false
      }
    }

    return {
      selectedFile,
      fileFormat,
      dragOver,
      uploading,
      uploadError,
      uploadSuccess,
      handleFileSelect,
      handleDrop,
      removeFile,
      formatFileSize,
      uploadFile
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
  max-width: 600px;
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
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
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

.upload-body {
  padding: 2rem;
}

.format-selector {
  margin-bottom: 1.5rem;
}

.format-selector label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #374151;
}

.format-selector select {
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
}

.upload-zone {
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  margin-bottom: 1.5rem;
  transition: all 0.3s;
  background: #fafafa;
}

.upload-zone.drag-over {
  border-color: #10b981;
  background: #f0fdf4;
  transform: scale(1.02);
}

.upload-zone.has-file {
  border-color: #10b981;
  background: #f0fdf4;
}

.upload-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: 0.7;
}

.upload-placeholder p {
  margin: 0.5rem 0;
  color: #6b7280;
}

.file-info {
  font-size: 0.85rem;
  color: #9ca3af;
}

.file-selected {
  display: flex;
  align-items: center;
  gap: 1rem;
  text-align: left;
}

.file-icon {
  font-size: 2rem;
}

.file-details {
  flex: 1;
}

.file-details p {
  margin: 0.25rem 0;
}

.format-help {
  background: #f8fafc;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
}

.format-help h4 {
  margin: 0 0 1rem 0;
  color: #374151;
}

.code-example {
  background: #1f2937;
  color: #10b981;
  padding: 1rem;
  border-radius: 6px;
  font-size: 0.85rem;
  overflow-x: auto;
  margin: 0;
}

.upload-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn {
  background: #10b981;
  color: white;
}

.btn:hover:not(:disabled) {
  background: #059669;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.btn-secondary {
  background: #6b7280;
  color: white;
}

.btn-secondary:hover {
  background: #4b5563;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.error-banner {
  background: #fef2f2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  border: 1px solid #fecaca;
}

.success-banner {
  background: #f0fdf4;
  color: #166534;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  border: 1px solid #bbf7d0;
}

@media (max-width: 640px) {
  .upload-body {
    padding: 1.5rem;
  }
  
  .file-selected {
    flex-direction: column;
    text-align: center;
  }
  
  .upload-actions {
    flex-direction: column;
  }
}
</style>
