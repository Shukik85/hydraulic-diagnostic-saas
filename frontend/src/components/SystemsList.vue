<template>
    <div class="systems-container">
        <div class="systems-header">
            <h2>🔧 Гидравлические системы</h2>
            <div class="header-actions">
                <button class="btn" @click="createSystemAlert">
                    ➕ Создать систему
                </button>
                <button class="btn btn-secondary" @click="loadSystems">
                    🔄 Обновить
                </button>
            </div>
        </div>

        <div v-if="loading" class="loading">
            ⏳ Загрузка систем...
        </div>

        <div v-else-if="error" class="error-banner">
            ❌ Ошибка: {{ error }}
            <button class="btn btn-sm" @click="loadSystems">Повторить</button>
        </div>

        <div v-else-if="systems.length === 0" class="empty-state">
            <div class="empty-icon">📋</div>
            <h3>Систем пока нет</h3>
            <p>Создайте первую систему</p>
            <button class="btn" @click="createSystemAlert">Создать систему</button>
        </div>

        <div v-else class="systems-grid">
            <div v-for="system in systems" :key="system.id" class="system-card">
                <div class="card-header">
                    <h3>{{ system.name }}</h3>
                    <span class="status-badge" :class="`status-${system.status}`">
                        {{ system.status === 'active' ? '✅ Активна' :
                            system.status === 'maintenance' ? '🔧 Обслуживание' :
                                system.status === 'inactive' ? '⏸️ Неактивна' : '❌ Неисправна' }}
                    </span>
                </div>

                <div class="card-body">
                    <div class="system-info">
                        <p><strong>Тип:</strong> {{ system.system_type_display }}</p>
                        <p><strong>Местоположение:</strong> {{ system.location }}</p>
                        <p><strong>Макс. давление:</strong> {{ system.max_pressure }} бар</p>
                        <p><strong>Расход:</strong> {{ system.flow_rate }} л/мин</p>
                        <p><strong>Создана:</strong> {{ formatDate(system.created_at) }}</p>
                    </div>
                </div>

                <div class="card-actions">
                    <button class="btn btn-sm" @click="viewSystem(system)">
                        👁️ Подробнее
                    </button>
                    <button class="btn btn-sm btn-secondary" @click="editSystemAlert(system)">
                        ✏️ Редактировать
                    </button> <button class="btn btn-sm btn-secondary" @click="showUploadForm(system.id)">
                        📊 Данные
                    </button>

                    <button class="btn btn-sm" @click="diagnoseSystem(system.id)">
                        🔍 Диагностика
                    </button>
                    <button class="btn btn-sm btn-danger" @click="deleteSystem(system.id)">
                        🗑️ Удалить
                    </button>
                </div>
            </div>
        </div>
        <!-- МОДАЛЬНЫЕ ОКНА -->
        <SystemForm v-if="showCreateForm" @close="showCreateForm = false" @success="handleFormSuccess" />

        <SystemForm v-if="showEditForm && editingSystem" :system="editingSystem" @close="showEditForm = false"
            @success="handleFormSuccess" />

        <FileUpload v-if="showUploadDialog && uploadSystemId" :systemId="uploadSystemId"
            @close="showUploadDialog = false" @success="handleUploadSuccess" />
    </div>
</template>


<script>
import { ref, onMounted } from 'vue'
import { hydraulicSystemService } from '@/services/hydraulicSystemService'
import SystemForm from './SystemForm.vue'
import FileUpload from './FileUpload.vue'

export default {
    name: 'SystemsList',
    components: {
        SystemForm,
        FileUpload
    },
    setup() {
        const systems = ref([])
        const loading = ref(false)
        const error = ref(null)
        const showCreateForm = ref(false)
        const showEditForm = ref(false)
        const editingSystem = ref(null)
        const showUploadDialog = ref(false)
        const uploadSystemId = ref(null)

        const loadSystems = async () => {
            loading.value = true
            error.value = null
            try {
                const data = await hydraulicSystemService.getSystems()
                systems.value = Array.isArray(data) ? data : data.results || []
            } catch (err) {
                error.value = err.response?.data?.detail || err.message || 'Неизвестная ошибка'
                console.error('Load systems error:', err)
            } finally {
                loading.value = false
            }
        }

        const createSystemAlert = () => {
            showCreateForm.value = true  // ЗАМЕНИ alert на это
        }

        const editSystemAlert = (system) => {
            editingSystem.value = { ...system }
            showEditForm.value = true  // ЗАМЕНИ alert на это
        }

        const deleteSystem = async (systemId) => {
            const system = systems.value.find(s => s.id === systemId)
            if (confirm(`Удалить систему "${system?.name}"?`)) {
                try {
                    await hydraulicSystemService.deleteSystem(systemId)
                    await loadSystems()
                    alert('Система успешно удалена')
                } catch (err) {
                    error.value = 'Ошибка удаления системы'
                    console.error('Delete system error:', err)
                }
            }
        }

        const viewSystem = (system) => {
            alert(`Просмотр системы "${system.name}" - детали будут позже`)
        }

        const diagnoseSystem = async (systemId) => {
            try {
                await hydraulicSystemService.diagnoseSystem(systemId)
                alert('Диагностика запущена успешно!')
                await loadSystems()
            } catch (err) {
                error.value = 'Ошибка запуска диагностики'
                console.error('Diagnose system error:', err)
            }
        }

        const showUploadForm = (systemId) => {  // ДОБАВЬ ЭТОТ МЕТОД
            uploadSystemId.value = systemId
            showUploadDialog.value = true
        }

        const handleFormSuccess = () => {  // ДОБАВЬ ЭТОТ МЕТОД
            loadSystems()
            showCreateForm.value = false
            showEditForm.value = false
            editingSystem.value = null
        }

        const handleUploadSuccess = () => {  // ДОБАВЬ ЭТОТ МЕТОД
            loadSystems()
            showUploadDialog.value = false
            uploadSystemId.value = null
        }

        const formatDate = (dateString) => {
            return new Date(dateString).toLocaleString('ru-RU')
        }

        onMounted(() => {
            loadSystems()
        })

        return {
            systems,
            loading,
            error,
            loadSystems,
            createSystemAlert,
            editSystemAlert,
            deleteSystem,
            viewSystem,
            diagnoseSystem,
            formatDate,
            showCreateForm,     // ДОБАВЬ
            showEditForm,       // ДОБАВЬ
            editingSystem,      // ДОБАВЬ
            showUploadDialog,   // ДОБАВЬ
            uploadSystemId,     // ДОБАВЬ
            showUploadForm,     // ДОБАВЬ
            handleFormSuccess,  // ДОБАВЬ
            handleUploadSuccess // ДОБАВЬ
        }
    }
}
</script>

<style scoped>
.systems-container {
    padding: 2rem;
}

.systems-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.header-actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.loading,
.error-banner {
    text-align: center;
    padding: 2rem;
    border-radius: 8px;
    margin: 2rem 0;
}

.error-banner {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.systems-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 1.5rem;
}

.system-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.system-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.card-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-header h3 {
    margin: 0;
    font-size: 1.25rem;
}

.status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-active {
    background: rgba(34, 197, 94, 0.2);
    color: #16a34a;
}

.status-maintenance {
    background: rgba(245, 158, 11, 0.2);
    color: #d97706;
}

.status-inactive {
    background: rgba(107, 114, 128, 0.2);
    color: #6b7280;
}

.status-faulty {
    background: rgba(239, 68, 68, 0.2);
    color: #dc2626;
}

.card-body {
    padding: 1.5rem;
}

.system-info p {
    margin: 0.5rem 0;
    color: #4b5563;
}

.card-actions {
    display: flex;
    gap: 0.5rem;
    padding: 1rem 1.5rem;
    background: #f8fafc;
    border-top: 1px solid #e5e7eb;
    flex-wrap: wrap;
}

.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    text-decoration: none;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.875rem;
}

.btn:hover {
    background: #2563eb;
}

.btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 0.8rem;
}

.btn-secondary {
    background: #6b7280;
}

.btn-secondary:hover {
    background: #4b5563;
}

.btn-danger {
    background: #dc2626;
}

.btn-danger:hover {
    background: #b91c1c;
}

@media (max-width: 768px) {
    .systems-header {
        flex-direction: column;
        align-items: stretch;
    }

    .systems-grid {
        grid-template-columns: 1fr;
    }

    .card-actions {
        justify-content: center;
    }
}
</style>
