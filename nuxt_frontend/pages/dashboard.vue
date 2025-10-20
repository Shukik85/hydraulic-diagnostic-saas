<template>
  <div class="dashboard">
    <header class="dashboard-header">
      <h1>Панель управления гидравлическими системами</h1>
      <button @click="handleLogout" class="logout-btn">Выйти</button>
    </header>

    <div class="dashboard-content">
      <!-- Раздел списка систем -->
      <section class="systems-section">
        <div class="section-header">
          <h2>Список систем</h2>
          <button @click="showCreateForm" class="btn-primary">
            + Добавить систему
          </button>
        </div>
        <SystemsList
          @edit="handleEdit"
          @delete="handleDelete"
          @select="handleSystemSelect"
        />
      </section>
      <!-- Форма создания/редактирования системы -->
      <section v-if="showForm" class="form-section">
        <SystemForm
          :system="selectedSystem"
          @submit="handleFormSubmit"
          @cancel="handleFormCancel"
        />
      </section>
      <!-- Раздел загрузки файлов -->
      <section class="upload-section">
        <FileUpload />
      </section>
      <!-- Раздел отчетов -->
      <section v-if="selectedSystemForReports" class="reports-section">
        <ReportsList :system-id="selectedSystemForReports" />
      </section>
      <!-- RAG Assistant -->
      <section class="rag-assistant-section">
        <RagAssistant />
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

definePageMeta({
  middleware: 'auth'
})

const { logout } = useAuth()
const { fetchSystems, deleteSystem } = useSystems()

const showForm = ref(false)
const selectedSystem = ref(null)
const selectedSystemForReports = ref(null)

// Показать форму создания
const showCreateForm = () => {
  selectedSystem.value = null
  showForm.value = true
}

// Обработчик редактирования
const handleEdit = (system) => {
  selectedSystem.value = system
  showForm.value = true
}

// Обработчик удаления
const handleDelete = async (systemId) => {
  if (confirm('Вы уверены, что хотите удалить эту систему?')) {
    try {
      await deleteSystem(systemId)
      // Если удаляем выбранную систему, очищаем выбор
      if (selectedSystemForReports.value === systemId) {
        selectedSystemForReports.value = null
      }
      await fetchSystems()
    } catch (error) {
      console.error('Ошибка при удалении системы:', error)
    }
  }
}

// Обработчик выбора системы для просмотра отчетов
const handleSystemSelect = (systemId) => {
  selectedSystemForReports.value = systemId
}

// Обработчик отправки формы
const handleFormSubmit = async () => {
  showForm.value = false
  selectedSystem.value = null
  await fetchSystems()
}

// Обработчик отмены формы
const handleFormCancel = () => {
  showForm.value = false
  selectedSystem.value = null
}

// Обработчик выхода
const handleLogout = async () => {
  try {
    await logout()
    await navigateTo('/login')
  } catch (error) {
    console.error('Ошибка выхода:', error)
  }
}

// Загрузка систем при монтировании
onMounted(async () => {
  await fetchSystems()
})
</script>

<style scoped>
.dashboard {
  min-height: 100vh;
  background-color: #f5f5f5;
}

.dashboard-header {
  background: white;
  padding: 20px 40px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.dashboard-header h1 {
  margin: 0;
  color: #333;
  font-size: 24px;
}

.logout-btn {
  padding: 8px 16px;
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

.logout-btn:hover {
  background-color: #d32f2f;
}

.dashboard-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
}

.systems-section {
  grid-column: 1 / -1;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  margin: 0;
  color: #333;
  font-size: 20px;
}

.btn-primary {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

.btn-primary:hover {
  background-color: #45a049;
}

.form-section,
.upload-section,
.reports-section,
.rag-assistant-section {
  background: white;
  padding: 0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.reports-section,
.rag-assistant-section {
  grid-column: 1 / -1;
}

@media (max-width: 1024px) {
  .dashboard-content {
    grid-template-columns: 1fr;
  }

  .form-section,
  .upload-section,
  .reports-section,
  .rag-assistant-section {
    grid-column: 1;
  }
}

@media (max-width: 768px) {
  .dashboard-header {
    padding: 15px 20px;
    flex-direction: column;
    gap: 15px;
  }

  .dashboard-header h1 {
    font-size: 20px;
  }

  .section-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
}
</style>
